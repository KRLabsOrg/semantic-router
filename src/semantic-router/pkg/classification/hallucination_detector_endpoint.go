package classification

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// openai_compatible backend: the detector is a fine-tuned generative span model
// (e.g. KRLabsOrg/lettucedect-v2-qwen-2b) served behind any OpenAI-compatible
// endpoint, typically vLLM. One call returns typed spans (category/subcategory,
// optional explanation), replacing both the token-level detector and the NLI
// explainer. The backend/endpoint config shape mirrors the embedding provider
// support (EmbeddingBackendOpenAICompatible / EmbeddingEndpointConfig).
//
// The prompt wording below is the FROZEN training prompt of the lettucedect-v2
// generative detectors (lettucedetect/prompts/generative.py) — do not edit it
// without retraining the models.

// HallucinationBackendOpenAICompatible selects the remote generative detector backend.
const HallucinationBackendOpenAICompatible = "openai_compatible"

const defaultEndpointTimeout = 60 * time.Second

type taxonomyLabel struct{ name, description string }

// Order is part of the frozen training prompt — do not reorder.
var generativeCategories = []taxonomyLabel{
	{"contradiction", "conflicts with the context (a wrong value, number, date, name, or relationship)"},
	{"fabricated_reference", "an entity, name, identifier, or section that is absent from the context"},
	{"unsupported_addition", "a claim, detail, or behavior the context never states"},
}

var generativeSubcategories = []taxonomyLabel{
	{"entity", "a wrong or invented name, entity, or object"},
	{"temporal", "an incorrect date, time, duration, or ordering"},
	{"numerical", "an incorrect number, quantity, or amount"},
	{"value", "a wrong value, setting, or attribute value"},
	{"relational", "an incorrect relationship or association between things"},
	{"identifier", "an invented identifier or name not found in the context"},
	{"section", "a reference to a section, part, or location that does not exist"},
	{"attribute", "an invented or incorrect attribute or property"},
	{"claim", "an added factual claim the context does not support"},
	{"behavior", "an added or changed action or behavior the context never states"},
	{"elaboration", "extra detail or elaboration beyond what the context supports"},
	{"subjective", "an unsupported subjective or evaluative statement"},
	{"unspecified", "unsupported, with no more specific subtype"},
}

// severity maps taxonomy categories onto the existing 0-4 severity scale used
// by the NLI explainer path, so downstream warning shaping needs no changes.
var categorySeverity = map[string]int{
	"contradiction":        4,
	"fabricated_reference": 3,
	"unsupported_addition": 2,
}

func labelNames(labels []taxonomyLabel) []string {
	names := make([]string, len(labels))
	for i, l := range labels {
		names[i] = l.name
	}
	return names
}

func buildGenerativeSystemPrompt(explain bool) string {
	var cats, subs []string
	for _, l := range generativeCategories {
		cats = append(cats, fmt.Sprintf("- %s: %s", l.name, l.description))
	}
	for _, l := range generativeSubcategories {
		subs = append(subs, fmt.Sprintf("- %s: %s", l.name, l.description))
	}
	explClause, explField := "", ""
	if explain {
		explClause = ", and give a short explanation of why it is unsupported"
		explField = `, "explanation": "..."`
	}
	return "You are an expert annotator who identifies hallucinated spans in a generated answer " +
		"with respect to a given context (the only trusted evidence). A hallucinated span is a " +
		"substring of the answer that is not supported by the context. Spans consistent with the " +
		"context are not hallucinations.\n\n" +
		"Quote each hallucinated span verbatim from the answer and classify it into exactly one " +
		fmt.Sprintf("category and one subcategory%s.\n\n", explClause) +
		fmt.Sprintf("Categories (the kinds of unsupported span):\n%s\n\n", strings.Join(cats, "\n")) +
		fmt.Sprintf("Subcategories:\n%s\n\n", strings.Join(subs, "\n")) +
		"Reply with ONLY a JSON object (no markdown, no code fences): " +
		fmt.Sprintf(`{"hallucinated_spans": [{"text": "...", "category": "...", "subcategory": "..."%s}]}. `, explField) +
		`If nothing is unsupported, reply {"hallucinated_spans": []}.`
}

func buildGenerativeSchema(explain bool) map[string]any {
	properties := map[string]any{
		"text":        map[string]any{"type": "string"},
		"category":    map[string]any{"type": "string", "enum": labelNames(generativeCategories)},
		"subcategory": map[string]any{"type": "string", "enum": labelNames(generativeSubcategories)},
	}
	if explain {
		properties["explanation"] = map[string]any{"type": "string"}
	}
	required := make([]string, 0, len(properties))
	for k := range properties {
		required = append(required, k)
	}
	sort.Strings(required)
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"hallucinated_spans": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type":                 "object",
					"properties":           properties,
					"required":             required,
					"additionalProperties": false,
				},
			},
		},
		"required":             []string{"hallucinated_spans"},
		"additionalProperties": false,
	}
}

type generativeSpan struct {
	Text        string `json:"text"`
	Category    string `json:"category"`
	Subcategory string `json:"subcategory"`
	Explanation string `json:"explanation,omitempty"`
}

// IsEndpointBackend reports whether this detector calls an external generative
// model instead of the in-process candle token classifier.
func (d *HallucinationDetector) IsEndpointBackend() bool {
	return d.config.Backend == HallucinationBackendOpenAICompatible
}

// DetectViaEndpoint runs one generative detection call and returns typed spans.
// Category/subcategory land in the NLI-shaped fields (NLILabelStr, Explanation)
// so the existing response-warning pipeline works unchanged.
func (d *HallucinationDetector) DetectViaEndpoint(context, question, answer string) (*EnhancedHallucinationResult, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if !d.initialized {
		return nil, fmt.Errorf("hallucination detector not initialized")
	}
	return d.detectViaEndpointLocked(context, question, answer)
}

// detectViaEndpointLocked is DetectViaEndpoint without locking; callers must
// hold d.mu (read) and have checked d.initialized.
func (d *HallucinationDetector) detectViaEndpointLocked(context, question, answer string) (*EnhancedHallucinationResult, error) {
	if answer == "" {
		return &EnhancedHallucinationResult{HallucinationDetected: false, Confidence: 1.0}, nil
	}
	if context == "" {
		return nil, fmt.Errorf("context is required for hallucination detection")
	}

	explain := d.config.IncludeExplanation
	// Training serialization: request, context excerpts, then the answer to verify.
	userMsg := fmt.Sprintf("User request: %s\n\nExcerpt 1:\n%s\n\nAnswer to verify:\n%s", question, context, answer)
	payload := map[string]any{
		"model":       d.config.ModelID,
		"temperature": 0,
		"messages": []map[string]string{
			{"role": "system", "content": buildGenerativeSystemPrompt(explain)},
			{"role": "user", "content": userMsg},
		},
		"response_format": map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "hallucination_detection",
				"schema": buildGenerativeSchema(explain),
				"strict": true,
			},
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal detection request: %w", err)
	}

	timeout := defaultEndpointTimeout
	if d.config.Endpoint.TimeoutSeconds > 0 {
		timeout = time.Duration(d.config.Endpoint.TimeoutSeconds) * time.Second
	}
	url := strings.TrimSuffix(d.config.Endpoint.BaseURL, "/") + "/chat/completions"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build detection request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if d.config.Endpoint.APIKeyEnv != "" {
		if key := os.Getenv(d.config.Endpoint.APIKeyEnv); key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
	}
	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("hallucination endpoint request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("hallucination endpoint returned status %d", resp.StatusCode)
	}

	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&completion); err != nil {
		return nil, fmt.Errorf("decode endpoint response: %w", err)
	}
	if len(completion.Choices) == 0 {
		return nil, fmt.Errorf("endpoint response has no choices")
	}

	var parsed struct {
		HallucinatedSpans []generativeSpan `json:"hallucinated_spans"`
	}
	if err := json.Unmarshal([]byte(completion.Choices[0].Message.Content), &parsed); err != nil {
		return nil, fmt.Errorf("parse hallucinated_spans JSON: %w", err)
	}

	result := &EnhancedHallucinationResult{Spans: []EnhancedHallucinationSpan{}}
	used := [][2]int{} // claimed answer regions, for non-overlapping verbatim matching
	for _, sp := range parsed.HallucinatedSpans {
		if sp.Text == "" {
			continue
		}
		start := findNonOverlapping(answer, sp.Text, used)
		if start < 0 {
			logging.Debugf("Endpoint span not found verbatim in answer, dropping: %q", sp.Text)
			continue
		}
		end := start + len(sp.Text)
		used = append(used, [2]int{start, end})

		explanation := sp.Explanation
		if explanation == "" {
			explanation = fmt.Sprintf("%s/%s", sp.Category, sp.Subcategory)
		}
		nliLabel := NLINeutral
		if sp.Category == "contradiction" {
			nliLabel = NLIContradiction
		}
		result.Spans = append(result.Spans, EnhancedHallucinationSpan{
			Text:                    sp.Text,
			Start:                   start,
			End:                     end,
			HallucinationConfidence: 1.0, // the generative detector emits decisions, not scores
			NLILabel:                nliLabel,
			NLILabelStr:             fmt.Sprintf("%s/%s", sp.Category, sp.Subcategory),
			NLIConfidence:           1.0,
			Severity:                categorySeverity[sp.Category],
			Explanation:             explanation,
		})
	}
	result.HallucinationDetected = len(result.Spans) > 0
	if result.HallucinationDetected {
		result.Confidence = 1.0
	}

	logging.Debugf("Hallucination detection (endpoint): detected=%v, spans=%d",
		result.HallucinationDetected, len(result.Spans))
	return result, nil
}

// findNonOverlapping returns the first verbatim occurrence of sub in s that does
// not overlap an already-claimed region, or -1.
func findNonOverlapping(s, sub string, used [][2]int) int {
	from := 0
	for {
		idx := strings.Index(s[from:], sub)
		if idx < 0 {
			return -1
		}
		start := from + idx
		end := start + len(sub)
		overlaps := false
		for _, u := range used {
			if start < u[1] && u[0] < end {
				overlaps = true
				break
			}
		}
		if !overlaps {
			return start
		}
		from = start + 1
	}
}

// basicResultFromEnhanced downconverts an endpoint result for callers of the
// basic Detect API.
func basicResultFromEnhanced(enhanced *EnhancedHallucinationResult) *HallucinationResult {
	basic := &HallucinationResult{
		HallucinationDetected: enhanced.HallucinationDetected,
		Confidence:            enhanced.Confidence,
		UnsupportedSpans:      []string{},
		SupportedSpans:        []string{},
	}
	for _, span := range enhanced.Spans {
		basic.UnsupportedSpans = append(basic.UnsupportedSpans, span.Text)
	}
	return basic
}
