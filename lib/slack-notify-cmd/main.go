package main

import (
	"bytes"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/joho/godotenv"
)

func init() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}
}

func main() {
	name := "報告者"
	text := "順調じゃよ！"
	channel := os.Getenv("CHANNEL")

	jsonStr := `{"channel":"` + channel + `","username":"` + name + `","text":"` + text + `","icon_emoji":":ghost:"}"`

	req, err := http.NewRequest(
		"POST",
		os.Getenv("WEBHOOK"),
		bytes.NewBuffer([]byte(jsonStr)),
	)

	fmt.Println(jsonStr)

	if err != nil {
		fmt.Print(err)
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Print(err)
	}

	fmt.Print(resp)
	defer resp.Body.Close()
}
