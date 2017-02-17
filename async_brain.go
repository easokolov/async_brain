package main

// Online POC version
// https://play.golang.org/p/lA0yztEaOP

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync" // Mutex
)

var debug int32

func _sigmoid_(f float64) float64 {
	/*
		// Сигмоидальная переходная функция нейрона
		// В размерности float имеет смысл вычислять sigmoid только для промежутка от -14 до +15 (При T=0, A=1).
		// Значение сигмоиды при этом проходит от 0 до 1.
		// При аргументе от -4 до 5 сигмоида пробегает от 0.01(7) до 0.99(3).
		// При аргументе от -7 до 7 сигмоида пробегает от 0.000911 до 0.9990889.
		// В формуле ( 1 / (1 + expf(T - A * zz)) )    Величина T влияет на смещение результатов сигмоиды по абсциссе. Величина A "сплющивает" или "растягиавет" сигмоиду.
	*/
	return (1.0 / (1.0 + math.Exp(-2.7*f)))
}

/*
!!!!!!!!!!!!!!!!!!!!!!!!!
Можно сделать, чтобы выходным тоже был слайс каналов. При этом, нейрон - это может быть структура, а вышеописанная функция - это может быть ее метод.
А также будет другой метод: "слушающий нейрон запрашивает у другого через этот метод выводной канал".
Этот нейрон создает новый канал и добавляет его в свой слайс выходных каналов.
А вопрошающий добавляет себе этот канал в слайс входных каналов (т.е. уже только указатель на него).
"Отдающий" нейрон при смене свонего значения пуляет его во все каналы из слайса выходных.
!!!!!!!!!!!!!!!!!!!!!!!!!
*/

type Neuron struct {
	ins    []<-chan float64 // Массив входных каналов нейрона (здесь только указатели на них. Создаёт их выдающий сигнал нейрон).
	in     []float64        // Кэш значений входных синапсов
	inmux  sync.Mutex       // Мьютекс для работы с входными значениями
	weight []float64        // веса. Размерность та же
	out    float64          // кэш выходного значения
	outs   []chan<- float64 // Массив каналов. По всем ним передается значение out для разных получателей.
	outmux sync.Mutex       // Мьютекс для работы с выходными значениями
	// Канал добавляется, когда другой нейрон запрашивает у этого связь с ним.
}

func (N *Neuron) make_link() chan float64 {
	c := make(chan float64, 1)
	N.outs = append(N.outs, c)
	return c
}

// Связать нейрон N c нейроном N2 с весом W
// N2 будет испускать сигналы, а N будет их обрабатывать.
// N Запрашивает у N2 создание канала.
func (N *Neuron) link_with(N2 *Neuron, W float64) {
	N.ins = append(N.ins, N2.make_link())
	N.in = append(N.in, 0.0)
	N.weight = append(N.weight, W)
}

func (N *Neuron) calc() {
	val := 0.0
	for i, v := range N.in {
		val += v * N.weight[i]
	}
	val = _sigmoid_(val)
	delta := math.Abs((N.out - val) / N.out)
	N.out = val
	//if debug == 1 {
	//	fmt.Println("Neuron:", N, "returns value:", val)
	//}
	if delta > 0.001 { // Если значение изменилось не больше, чем на 0.1%, то сигнал не подаем.
		//FIXME
		//fmt.Println(N.out, ">>", N.outs)
		for _, c := range N.outs {
			fmt.Println("\t\t\t\t\t\t\t\tSending", val, "to", c, "of [", N.outs, "]")
			go func(cc chan<- float64, value float64) {
				fmt.Println("Sending", value, "into", cc)
				cc <- value
			}(c, val)
			//c <- val
		}
	} else {
		fmt.Println("!!!!!!!!!!!!!!! delta is too low.", N.out, "wouldn't be sent to", N.outs)
	}
}

func (N *Neuron) listen() {
	N.inmux.Lock()
	defer N.inmux.Unlock()
	for i, syn := range N.ins {
		go func(NN *Neuron, n int, s <-chan float64) { //FIXME Поведение отличается в зависимости от того 'go func' или просто 'func'. Видимо, надо разбираться с замыканиями.
			// Выяснили, что в этой функции использовать 'i' и 'syn' нельзя, а можно только переданные их значения 'n' и 's'.
			//fmt.Println("###", NN, "START LISTENING [", i, "]", syn, "[", n, "]", s) // видим, что i и n различаются.
			for {
				//fmt.Println(NN.ins, "[", n, "]", ">>to>>", NN.outs) // !!!!!!!!!!!!
				select {
				case NN.in[n] = <-s:
					//FIXME
					//if debug == 1 {
					//	fmt.Println("chanel", s, "gotcha in select. n =", n, "Value received:", NN.in[n])
					//}
					go NN.calc()
				}
			}
		}(N, i, syn)
	}
}

func main() {
	// Building NN
	/*
		n1 := Neuron{
			make([]<-chan float64, 0, 4),
			make([]float64, 0, 4),
			make([]float64, 0, 4),
			0.0,
			make([]chan<- float64, 0, 4),
		}
		n2 := Neuron{
			make([]<-chan float64, 0, 4),
			make([]float64, 0, 4),
			make([]float64, 0, 4),
			0.0,
			make([]chan<- float64, 0, 4),
		}
	*/
	var N [4]Neuron

	//N[0].link_with(&N[1], 0.9)
	////N[0].link_with(&N[2], -0.9)
	////N[1].link_with(&N[0], 0.666)
	//N[1].link_with(&N[2], 0.666)
	//N[2].link_with(&N[0], 0.55)
	//N[2].link_with(&N[1], -0.7)

	/**/
	N[2].link_with(&N[2], 0.12)
	N[2].link_with(&N[1], 1.09)
	N[3].link_with(&N[0], -2.2)
	N[1].link_with(&N[3], 0.1)
	N[0].link_with(&N[2], -0.54)
	N[3].link_with(&N[1], -1.22)
	N[0].link_with(&N[3], 0.87)
	N[1].link_with(&N[2], -0.32)
	N[3].link_with(&N[2], 0.89)
	/**/
	// End of building NN

	// Запуск
	debug = 1
	N[0].outs[0] <- 1.0

	// Работа.
	// Можно `go N[x].lisen()`. Но и так получаются отдельные треды (запускаются в самой ф-ции listen)
	N[0].listen()
	N[1].listen()
	N[2].listen()
	N[3].listen()
	for {
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("Enter text: ")
		text, _ := reader.ReadString('\n')
		if text[0] == "q"[0] {
			return
		}
		in_int, err := strconv.ParseFloat(strings.Split(text, "\n")[0], 64)
		if err == nil {
			fmt.Println("_sigmoid_(", in_int, "):", _sigmoid_(in_int))
			//FIXME
			//if in_int == -1 {
			debug = 1
			//}
			N[0].outs[0] <- in_int
		}
	}
}
