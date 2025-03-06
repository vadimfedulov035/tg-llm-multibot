package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"io"
	"os"
	"strings"
	"time"
)

// Server specific constant
const (
	API          = "http://llm-server:8000/v1/generate"
	MAX_SEND_TRY = 3
	RETRY_TIME   = 5 * time.Second
	API_TIMEOUT  = 10 * time.Minute
)

// Settings JSON representation
type Settings struct {
	SystemPrompt string   `json:"system_prompt"`
	ChainPrompts []string `json:"chain_prompts"`
	RatePrompt   string   `json:"rate_prompt"`

	Temperature       float32 `json:"temperature"`
	RepetitionPenalty float32 `json:"repetition_penalty"`
	TopP              float32 `json:"top_p"`
	TopK              int     `json:"top_k"`

	RespTokens     int `json:"response_tokens"`
	RespTokenShift int `json:"response_token_shift"`
	RespBatchSize  int `json:"response_batch_size"`

	RateTokens    int `json:"rate_tokens"`
	RateBatchSize int `json:"rate_batch_size"`
}

// Request to send
type RequestBody struct {
	Dialog   []string `json:"dialog"`
	Settings Settings `json:"settings"`
}

// Response to receive
type ResponseBody struct {
	Response string `json:"response"`
}

func loadSettings(config string) Settings {
	var settings Settings

	// Read JSON data from file
	data, err := os.ReadFile(config)
	if err != nil {
		log.Fatalf("[OS] Failed to read settings file: %s", err)
	}

	// Decode JSON data to settings
	err = json.Unmarshal(data, &settings)
	if err != nil {
		log.Fatalf("[OS] Failed to unmarshal settings: %s", err)
	}
	return settings
}

// Request constructor
func newRequestBody(dialog []string, config string) *RequestBody {
	// Return request body: dialog, settings
	return &RequestBody{
		Dialog:   dialog,
		Settings: loadSettings(config),
	}
}

func sendRequestBody(requestBody *RequestBody) (string, error) {
	// Encode request body to JSON data
	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return "", err
	}

	// Make new POST request with JSON data
	request, err := http.NewRequest("POST", API, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("[API] Failed to make request: %s", err)
	}
	request.Header.Set("Content-Type", "application/json")

	// Set HTTP client
	client := &http.Client{Timeout: API_TIMEOUT}
	resp, err := client.Do(request)
	if err != nil {
		return "", fmt.Errorf("[API] Failed to send request: %s", err)
	}
	defer resp.Body.Close()

	// Check status; print status code of error if any
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Status %d: %s", resp.StatusCode, string(body))
	}

	// Decode response body
	var responseBody ResponseBody
	err = json.NewDecoder(resp.Body).Decode(&responseBody)
	if err != nil {
		return "", err
	}

	return responseBody.Response, nil
}

// Send request to LLM server
func Send(dialog []string, config string, chatTitle string) (string, error) {
	// Prepare request body
	requestBody := newRequestBody(dialog, config)

	// Add chat title to system prompt if space reserved
	prompt := requestBody.Settings.SystemPrompt
	if strings.Count(prompt, "%s") != 1 {  
		errMsg := "[OS] No/many %%s in system prompt. Use one for chat title."
		return "", fmt.Errorf(errMsg)  
	}  
	if strings.Contains(prompt, "%s") {
		prompt = fmt.Sprintf(prompt, chatTitle)
	}
	requestBody.Settings.SystemPrompt = prompt

	// Send request body (<MAX_SEND_TRY> tries)
	var text string
	var err error
	for i := range MAX_SEND_TRY {
		text, err = sendRequestBody(requestBody)
		if err == nil {
			break
		}
		log.Printf("[API] Try %d: %v", i, err)
		time.Sleep(RETRY_TIME * (1 << (i+1)))
	}

	return text, err
}
