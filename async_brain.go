package main

import "fmt"

//import "strings"

func main() {
	in := make(chan string)
	mid := make(chan string)
	out := make(chan string)
	str := "zzzzzz"

	go pinger(in, mid)
	go ponger(mid, out)
	for i := 0; i < 10; i++ {
		in <- str
		str = <-out
		str = "x" + str + "x"
		fmt.Println("inter", i, ":\t", str)
	}
}

func pinger(in <-chan string, out chan<- string) {
	for {
		//str := strings.Split(<-in, "@")[1]
		str := "|" + <-in + "|"
		fmt.Println("PINGER:\t\t", str)
		out <- str
	}
}

func ponger(in <-chan string, out chan<- string) {
	for {
		str := "@" + <-in + "@"
		fmt.Println("PONGER:\t\t", str)
		out <- str
	}
}

func neur() <-chan float64 {

}
