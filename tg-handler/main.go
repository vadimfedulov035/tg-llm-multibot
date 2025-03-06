package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"os/signal"
	"syscall"
	"path/filepath"
	"sync"

	tg "github.com/go-telegram-bot-api/telegram-bot-api/v5"

	"tg-handler/api"
	"tg-handler/memory"
	"tg-handler/messaging"
)

// Path, structure and loader for Initial Config
const InitConf = "./confs/init.json"

type InitJSON struct {
	KeysAPI     []string            `json:"keysAPI"`
	Admins      []string            `json:"admins"`
	Orders      map[string][]string `json:"orders"`
	ConfigPath  string              `json:"config_path"`
	HistoryPath string              `json:"history_path"`
	MemoryLimit int                 `json:"memory_limit"`
}

func loadInitJSON(config string) *InitJSON {
	var initJSON InitJSON

	// Read JSON data from file
	data, err := os.ReadFile(config)
	if err != nil {
		log.Fatalf("Failed to read %s: %v", InitConf, err)
	}

	// Decode JSON data to InitJSON
	err = json.Unmarshal(data, &initJSON)
	if err != nil {
		log.Fatalf("Failed to unmarshal %s: %v", InitConf, err)
	}

	return &initJSON
}

func handleMessage(c *messaging.ChatInfo, sh *memory.SafeChatHistory) {
	// Type until reply
	ctx, cancel := context.WithCancel(context.Background())
	go messaging.Typing(ctx, c)
	defer cancel()

	// Add old message pair to history, get it (as interfaces)
	m := messaging.NewMessageInfo(c.Bot, c.Message.ReplyToMessage)
	lines := memory.Add([2]memory.IMessage{c, m}, "", sh)

	// Get dialog, send to API and reply
	dialog := memory.Get(lines, sh, c.MemoryLimit)
	text, err := api.Send(dialog, c.Config, c.ChatTitle)
	if err != nil {
		log.Printf("API error in chat %s.", c.ChatTitle)
		return
	}
	resp := messaging.Reply(c, text)

	// Add new message pair to history (last: interface, previous: reused)
	m = messaging.NewMessageInfo(c.Bot, resp)
	memory.Add([2]memory.IMessage{m, nil}, lines[0], sh)
}

func botHandler(i int, initJSON *InitJSON, safeHistory *memory.SafeHistory) {
	// Start bot from specific keyAPI
	keysAPI := initJSON.KeysAPI
	bot, err := tg.NewBotAPI(keysAPI[i])
	if err != nil {
		panic(err)
	}
	// Log authorization
	botName := bot.Self.UserName
	log.Printf("Authorized on account %s", botName)

	// Get constants
	Admins := initJSON.Admins
	Orders := initJSON.Orders[botName]
	MemoryLimit := initJSON.MemoryLimit
	HistoryPath := initJSON.HistoryPath
	ConfigPath := initJSON.ConfigPath

	// Get bot history and config (order postfix added by OrderInfo)
	safeBotHistory := safeHistory.Get(botName)
	botConfig := filepath.Join(ConfigPath, botName+"%s.json")

	// Start update channel
	u := tg.NewUpdate(0)
	u.Timeout = 30
	updates := bot.GetUpdatesChan(u)
	for update := range updates {
		message := update.Message

		// Validate the message (messageInfo: Bot, Message, Text, Sender)
		messageInfo := messaging.NewMessageInfo(bot, message)
		isNil := messageInfo == nil
		isValid := messageInfo.Text != "" || messageInfo.Sender != ""
		if isNil || !isValid {
			continue
		}

		// Identify the message (OrderInfo: Config, Order)
		orderInfo := messaging.NewOrderInfo(messageInfo, botConfig, Orders)
		isAsked := messaging.IsAsked(orderInfo, Admins)
		if !isAsked {
			continue
		}
		log.Printf("%s got message", botName)

		// Get chat history (ChatInfo: CID, ChatTitle)
		chatInfo := messaging.NewChatInfo(orderInfo, MemoryLimit)
		safeChatHistory := safeBotHistory.Get(chatInfo.CID)

		// Handle the message
		handleMessage(chatInfo, safeChatHistory)

		// Clean and save history
		memory.CleanHistory(safeHistory)
		if err := memory.SaveHistory(HistoryPath, safeHistory); err != nil {  
			log.Printf("Failed to save history for %s: %v", botName, err)  
		}  
	}
}

func main() {
	// Terminate on termination signal gracefully
	ctx, cancel := signal.NotifyContext(
		context.Background(), os.Interrupt, syscall.SIGTERM,
	)
	defer cancel()

	// Load initialization config
	initJSON := loadInitJSON(InitConf)

	// Get KeysAPI and HistoryPath
	KeysAPI := initJSON.KeysAPI
	HistoryPath := initJSON.HistoryPath

	// Make safe history container
	history := memory.LoadHistory(HistoryPath)
	mu := new(sync.RWMutex)
	safeHistory := memory.NewSafeHistory(history, mu)

	// Start all bots with shared history and mutex
	for i := range KeysAPI {
		go botHandler(i, initJSON, safeHistory)
	}

	// Block until termination signal
	<-ctx.Done()
	log.Println("Shutting down...")  
}
