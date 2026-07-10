package classification

import (
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func newEndpointDetector(t *testing.T, url string) *HallucinationDetector {
	t.Helper()
	d, err := NewHallucinationDetector(&config.HallucinationModelConfig{
		ModelID:  "KRLabsOrg/lettucedect-v2-qwen-2b",
		Backend:  HallucinationBackendOpenAICompatible,
		Endpoint: config.HallucinationEndpointConfig{BaseURL: url},
	})
	if err != nil {
		t.Fatalf("NewHallucinationDetector: %v", err)
	}
	if err := d.Initialize(); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	return d
}

// The system prompt is the frozen lettucedect-v2 training prompt
// (lettucedetect/prompts/generative.py, verified byte-identical to the SFT
// training data) — any drift silently degrades the detector.
func TestFrozenSystemPromptHash(t *testing.T) {
	sum := sha1.Sum([]byte(buildGenerativeSystemPrompt(false)))
	if got := hex.EncodeToString(sum[:])[:12]; got != "cf5d3f3f3a9d" {
		t.Fatalf("frozen system prompt drifted: sha1 %s", got)
	}
}

func chatCompletion(content string) map[string]any {
	return map[string]any{
		"choices": []map[string]any{
			{"message": map[string]any{"content": content}},
		},
	}
}

func TestDetectViaEndpoint_TypedSpans(t *testing.T) {
	answer := "The Eiffel Tower is 450 metres tall and was designed by Leonardo."
	spansJSON := `{"hallucinated_spans": [` +
		`{"text": "450 metres", "category": "contradiction", "subcategory": "numerical"},` +
		`{"text": "designed by Leonardo", "category": "unsupported_addition", "subcategory": "claim"}]}`

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req["model"] != "KRLabsOrg/lettucedect-v2-qwen-2b" {
			t.Errorf("unexpected model %v", req["model"])
		}
		if _, ok := req["response_format"]; !ok {
			t.Error("request missing response_format")
		}
		_ = json.NewEncoder(w).Encode(chatCompletion(spansJSON))
	}))
	defer server.Close()

	d := newEndpointDetector(t, server.URL)
	result, err := d.DetectViaEndpoint("The Eiffel Tower is 330 metres tall.", "How tall is it?", answer)
	if err != nil {
		t.Fatalf("DetectViaEndpoint: %v", err)
	}
	if !result.HallucinationDetected || len(result.Spans) != 2 {
		t.Fatalf("expected 2 spans detected, got detected=%v spans=%d",
			result.HallucinationDetected, len(result.Spans))
	}

	first := result.Spans[0]
	if first.Text != "450 metres" || first.Start != 20 || first.End != 30 {
		t.Errorf("bad offsets: %+v", first)
	}
	if first.NLILabelStr != "contradiction/numerical" || first.Severity != 4 || first.NLILabel != NLIContradiction {
		t.Errorf("bad taxonomy mapping: %+v", first)
	}
	if second := result.Spans[1]; second.Severity != 2 || second.NLILabel != NLINeutral {
		t.Errorf("bad taxonomy mapping: %+v", second)
	}

	// Basic API downconverts the same call.
	basic, err := d.Detect("The Eiffel Tower is 330 metres tall.", "How tall is it?", answer)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if !basic.HallucinationDetected || len(basic.UnsupportedSpans) != 2 {
		t.Errorf("basic downconversion wrong: %+v", basic)
	}
}

func TestDetectViaEndpoint_NoHallucination(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(chatCompletion(`{"hallucinated_spans": []}`))
	}))
	defer server.Close()

	d := newEndpointDetector(t, server.URL)
	result, err := d.DetectViaEndpoint("ctx", "q", "a faithful answer")
	if err != nil {
		t.Fatalf("DetectViaEndpoint: %v", err)
	}
	if result.HallucinationDetected || len(result.Spans) != 0 {
		t.Errorf("expected clean result, got %+v", result)
	}
}

func TestDetectViaEndpoint_SpanNotInAnswerDropped(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(chatCompletion(
			`{"hallucinated_spans": [{"text": "not present", "category": "contradiction", "subcategory": "value"}]}`))
	}))
	defer server.Close()

	d := newEndpointDetector(t, server.URL)
	result, err := d.DetectViaEndpoint("ctx", "q", "the actual answer")
	if err != nil {
		t.Fatalf("DetectViaEndpoint: %v", err)
	}
	if result.HallucinationDetected {
		t.Errorf("hallucinated span absent from answer must be dropped: %+v", result)
	}
}

func TestDetectViaEndpoint_EndpointDown(t *testing.T) {
	server := httptest.NewServer(nil)
	url := server.URL
	server.Close() // connection refused from here on

	d := newEndpointDetector(t, url)
	if _, err := d.DetectViaEndpoint("ctx", "q", "answer"); err == nil {
		t.Fatal("expected error when endpoint is unreachable")
	}
}

func TestEndpointBackend_RequiresEndpoint(t *testing.T) {
	d, err := NewHallucinationDetector(&config.HallucinationModelConfig{
		ModelID: "m",
		Backend: HallucinationBackendOpenAICompatible,
	})
	if err != nil {
		t.Fatalf("NewHallucinationDetector: %v", err)
	}
	if err := d.Initialize(); err == nil {
		t.Fatal("expected Initialize to fail without endpoint URL")
	}
}
