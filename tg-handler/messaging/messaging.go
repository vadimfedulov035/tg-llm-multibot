package messaging

import (
	"context"
	"fmt"
	"log"
	"slices"
	"strings"
	"time"

	tg "github.com/go-telegram-bot-api/telegram-bot-api/v5"
)

// Time interval for sending "Typing" signal to chat
const TypingInterval = 3

// Base type for message validation
type MessageInfo struct {
	Bot     *tg.BotAPI
	Message *tg.Message
	Sender  string
	Text    string
}

// Base type for check if bot is asked
type OrderInfo struct {
	MessageInfo
	Order  string
	Config string
}

// Base type for further LLM processing
type ChatInfo struct {
	OrderInfo
	CID         int64
	ChatTitle   string
	MemoryLimit int
}

// Constructors for checks and further processing
func NewMessageInfo(bot *tg.BotAPI, message *tg.Message) *MessageInfo {
	// nil message -> nil message info
	if message == nil {
		return nil
	}
	return &MessageInfo{
		Bot:     bot,
		Message: message,
		Sender:  getName(message),
		Text:    getText(bot, message),
	}
}

func NewOrderInfo(m *MessageInfo, config string, orders []string) *OrderInfo {
	text := m.Text

	// Get order and config to set
	order := getOrder(text, orders)
	orderConfig := getOrderConfig(config, order)

	return &OrderInfo{
		MessageInfo: *m,
		Order:       order,
		Config:      orderConfig,
	}
}

func NewChatInfo(o *OrderInfo, memoryLimit int) *ChatInfo {
	message := o.Message
	sender := o.Sender

	return &ChatInfo{
		OrderInfo:  *o,
		CID:         getCID(message),
		ChatTitle:   getChatTitle(message, sender),
		MemoryLimit: memoryLimit,
	}
}

// Base getters to comply with IMessage interface from memory module
func (m *MessageInfo) GetText() string {
	if m != nil {
		return m.Text
	}
	return ""
}

func (m *MessageInfo) GetSender() string {
	if m != nil {
		return m.Sender
	}
	return ""
}

func (m *MessageInfo) GetOrder() string { // Comply with empty order getter
	return ""
}

func (r *OrderInfo) GetOrder() string { // Reimplement order getter
	if r != nil {
		return r.Order
	}
	return ""
}

// Get text from message (always humanized)
func getText(bot *tg.BotAPI, message *tg.Message) string {
	if message == nil {
		return ""
	}

	// Get any text
	var text string
	if message.Text != "" {
		text = message.Text
	} else if message.Caption != "" {
		text = message.Caption
	}

	// Substitute @name_bot to BotName for humanized style
	// Both form of addressing are detectable but only humanized will be passed
	humanize := func(text string, self *tg.User) string {
		botName, botFirstName := self.UserName, self.FirstName
		text = strings.ReplaceAll(text, "@"+botName, botFirstName+",")
		return text
	}
	text = humanize(text, &bot.Self)

	return text
}

// Get name if any or "anonym"
func getName(message *tg.Message) string {
	var name string

	// bot -> first name, user -> user name
	if message.From.IsBot {
		name = message.From.FirstName
	} else {
		name = message.From.UserName
	}

	// anonymous user -> anonym
	if name == "" {
		name = "anonym"
	}

	return name
}

// Get Chat ID for public and private chats
func getCID(message *tg.Message) int64 {
	var cid int64
	if message.Chat != nil {
		cid = message.Chat.ID
	} else {
		cid = message.From.ID
	}
	return cid
}

// Get chat title if any or return "User's chat"
func getChatTitle(message *tg.Message, name string) string {
	chatName := message.Chat.Title
	if chatName == "" {
		chatName = fmt.Sprintf("%s's chat", name)
	}
	return chatName
}

// Gets order if any or return empty order
func getOrder(text string, orders []string) (order string) {
	for _, oneOrder := range orders {
		if strings.Contains(text, oneOrder) {
			order = oneOrder
			break
		}
	}
	return
}

// Add order to bot config as postfix to get order config
func getOrderConfig(botConfig string, order string) string {
	configPostfix := strings.Replace(order, "/", "_", 1)
	messageConfig := fmt.Sprintf(botConfig, configPostfix)

	return messageConfig
}

// Check if bot is asked (or ordered)
func IsAsked(c *OrderInfo, admins []string) bool {
	bot, message := c.Bot, c.Message
	text, sender := c.Text, c.Sender
	order := c.Order

	// Get chat variable
	chat := message.Chat

	// Get bot's first name and ID (via self reference)
	self := bot.Self
	botFirstName, botID := self.FirstName, self.ID

	// Get replied message if any for reply check
	replied := message.ReplyToMessage
	var repliedID int64 = 0
	if replied != nil {
		repliedID = replied.From.ID
	}

	// Chat status
	isPublic := chat.IsGroup() || chat.IsSuperGroup()
	isPrivate := chat.IsPrivate()

	// Bot reply conditions
	isReplied := repliedID == botID
	isMentioned := strings.Contains(text, botFirstName)
	isOrdered := order != ""
	isAdmin := slices.Contains(admins, sender)

	// Bot chat reply conditions
	isAskedPublicly := isPublic && (isReplied || isMentioned || isOrdered)
	isAskedPrivately := isPrivate && isAdmin

	// Bot ask status
	isAsked := isAskedPublicly || isAskedPrivately

	return isAsked
}

// Try to reply twice: with reply, with separate message
func Reply(c *ChatInfo, text string) *tg.Message {
	bot, message, cid := c.Bot, c.Message, c.CID

	// Construct message config and set up for reply
	messageConf := tg.NewMessage(cid, text)
	messageConf.ReplyToMessageID = message.MessageID

	// Try to reply with reply
	response, err := bot.Send(messageConf)
	if err != nil { // Try to reply with separate message
		messageConf.ReplyToMessageID = 0
		response, err = bot.Send(messageConf)
	}
	// Log final error
	if err != nil {
		log.Printf("[Telegram] Replying: %v", err)
	}

	return &response
}

// Send typing signal every <TypingInterval> seconds until context done
func Typing(ctx context.Context, c *ChatInfo) {
	bot, cid := c.Bot, c.CID

	// define function for sending typing signal
	sendTypingSignal := func() {
		actConf := tg.NewChatAction(cid, "typing")
		_, err := bot.Request(actConf)
		if err != nil {
			log.Printf("[Telegram] Action: %v", err)
		}
	}

	// send typing signal right away
	sendTypingSignal()

	// set <TypingInterval> second(s) ticker
	t := time.NewTicker(TypingInterval * time.Second)
	// stop on loop break
	defer t.Stop()

	// Send typing signal on every tick and break on context done
	for {
		select {
		case <-t.C:
			sendTypingSignal()
		case <-ctx.Done():
			return
		}
	}
}
