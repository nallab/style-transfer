package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/joho/godotenv"
)

var (
	name    = flag.String("n", "報告者", "Who is ?")
	message = flag.String("m", "順調じゃよ！", "Message")
)

func init() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}
}

func main() {
	flag.Parse()
	channel := os.Getenv("CHANNEL")

	jsonStr := `{"channel":"` + channel + `","username":"` + *name + `","text":"` + *message + `","icon_emoji":":ghost:"}"`

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
