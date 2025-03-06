package memory

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"io"
	"strings"
	"sync"
	"time"

	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

// Time limit for line deletion from history (change to 24 * time.Hour)
const TIME_LIMIT = 24 * time.Hour

// Message interface
type IMessage interface {
	GetText() string
	GetSender() string
	GetOrder() string
}

// Message type
type MessageEntry struct {
	Line      string    `json:"msg"`
	Timestamp time.Time `json:"ts"`
}

// History types
type (
	History     = map[string]BotHistory
	BotHistory  = map[int64]ChatHistory
	ChatHistory = map[string]MessageEntry
)

// Safe history containers (share the same mutex, as inference is consecutive)
type SafeHistory struct {
	History History
	mu      *sync.RWMutex
}
type SafeBotHistory struct {
	History BotHistory
	mu      *sync.RWMutex
}
type SafeChatHistory struct {
	History ChatHistory
	mu      *sync.RWMutex
}

// Safe history container constructors
func NewSafeHistory(hist History, mu *sync.RWMutex) *SafeHistory {
	return &SafeHistory{
		History: hist,
		mu:      mu,
	}
}

func NewSafeBotHistory(hist BotHistory, mu *sync.RWMutex) *SafeBotHistory {
	return &SafeBotHistory{
		History: hist,
		mu:      mu,
	}
}

func NewSafeChatHistory(hist ChatHistory, mu *sync.RWMutex) *SafeChatHistory {
	return &SafeChatHistory{
		History: hist,
		mu:      mu,
	}
}

// Safe history container getters
func (sh *SafeHistory) Get(botName string) *SafeBotHistory {
	// Ensure secure access
	sh.mu.Lock()
	defer sh.mu.Unlock()

	// Assign if nil
	if _, ok := sh.History[botName]; !ok {
		sh.History[botName] = make(BotHistory)
	}

	// Containerize
	safeBotHistory := NewSafeBotHistory(sh.History[botName], sh.mu)

	return safeBotHistory
}

func (sh *SafeBotHistory) Get(CID int64) *SafeChatHistory {
	// Ensure secure access
	sh.mu.Lock()
	defer sh.mu.Unlock()

	// Assign if nil
	if _, ok := sh.History[CID]; !ok {
		sh.History[CID] = make(ChatHistory)
	}

	// Containerize
	safeChatHistory := NewSafeChatHistory(sh.History[CID], sh.mu)

	return safeChatHistory
}

// Format message fields into line
func toLine(message IMessage) string {
	var result string

	text := message.GetText()
	sender := message.GetSender()
	order := message.GetOrder()

	// Empty text or sender -> empty line
	if text == "" || sender == "" {
		return ""
	}

	// Capitalize local function
	capitalize := func(str string) string {
		caser := cases.Title(language.English)
		strCap := caser.String(str)
		return strCap
	}

	// Order implies anonymous format "text" with stripped order
	// No order implies ordinary format "Name: text" with name capitalization
	if order != "" {
		if strings.HasSuffix(text, order) {
			result = strings.TrimSuffix(text, order)
		}
	} else {
		result = capitalize(sender) + ": " + text
	}

	return result
}

// Get lines via message interfaces or reused line
func getLines(messages [2]IMessage, line string) []string {
	// Convert last message to line and append (reverse order)
	lastLine := toLine(messages[0])
	lines := []string{lastLine}

	// Append reused line to lines (if passed)
	if line != "" {
		lines = append(lines, line)
		return lines
	}

	// Convert previous message to line and append (reverse order)
	prevLine := toLine(messages[1])
	if prevLine != "" {
		lines = append(lines, prevLine)
	}

	return lines
}

// Add lines pair to chat history, returns lines anyway
func Add(messages [2]IMessage, line string, sh *SafeChatHistory) []string {
	sh.mu.Lock()
	defer sh.mu.Unlock()

	// Get two lines or return
	lines := getLines(messages, line)
	if len(lines) < 2 {
		return lines
	}

	// Record two lines (last -> previous, as reverse order)
	sh.History[lines[0]] = MessageEntry{
		Line:      lines[1],
		Timestamp: time.Now(),
	}

	return lines
}

// Get dialog from chat history
func Get(lines []string, sh *SafeChatHistory, memoryLimit int) []string {
	sh.mu.RLock()
	defer sh.mu.RUnlock()

	// Got no pair to decipher
	if len(lines) < 2 {
		return lines
	}

	// Get two lines (reverse order)
	lastLine := lines[0]
	prevLine := lines[1]

	// Accumulate lines going backwards in history via reply chain
	lastLine = prevLine
	for i := range memoryLimit - 2 {
		if messageEntry, ok := sh.History[lastLine]; ok {
			log.Printf("%d messages remembered", i+1)

			prevLine = messageEntry.Line
			lines = append(lines, prevLine)
			lastLine = prevLine
		} else {
			break
		}
	}

	// Reverse lines in reverse order to get dialog
	reverse := func(lines []string) []string {
		for i, j := 0, len(lines)-1; i < j; i, j = i+1, j-1 {
			lines[i], lines[j] = lines[j], lines[i]
		}
		return lines
	}
	dialog := reverse(lines)

	return dialog
}

// Load history as shared once (no concurrency)
func LoadHistory(source string) History {
	var history History

	// Open file (created if needed)
	file, err := os.OpenFile(source, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		log.Fatalf("[OS error] Failed to open history file: %v", err)
	}
	defer file.Close()

	// Read JSON data from file
	data, err := io.ReadAll(file)
	if err != nil {
		log.Fatalf("[OS error] Failed to read history file: %v", err)
	}

	// Decode JSON data to history
	err = json.Unmarshal(data, &history)
	if err != nil {
		history = make(History)
		log.Println("[OS] History created")
	} else {
		log.Println("[OS] History loaded")
	}

	return history
}

// Save history concurrently
func SaveHistory(dest string, sh *SafeHistory) error {
	// Ensure secure access
	sh.mu.Lock()
	defer sh.mu.Unlock()

	// Open file
	file, err := os.OpenFile(dest, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return fmt.Errorf("[OS error] Failed to open history file: %v", err)
	}
	defer file.Close()

	// Encode history to JSON data
	data, err := json.Marshal(sh.History)
	if err != nil {
		return fmt.Errorf("[OS error] Failed to marshal history: %v", err)
	}

	// Write JSON data to file
	_, err = file.Write(data)
	if err != nil {
		return fmt.Errorf("[OS error] Failed to write history data: %v", err)
	}

	log.Println("[OS] History written")
	return nil
}

// Clean all lines older than day in every chat history
func CleanHistory(sh *SafeHistory) {
	// Ensure secure access
	sh.mu.Lock()
	defer sh.mu.Unlock()

	currentTime := time.Now()

	// Loop through all chat histories
	for _, botHistory := range sh.History {
		for _, chatHistory := range botHistory {
			// Initialize lines array
			var oldLines []string

			// Accumulate old lines
			for line, messageEntry := range chatHistory {
				if currentTime.Sub(messageEntry.Timestamp) > TIME_LIMIT {
					oldLines = append(oldLines, line)
				}
			}

			// Delete all old lines
			for _, line := range oldLines {
				delete(chatHistory, line)
			}
		}
	}
}
