package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

func _sigmoid_(f float64) float64 {
	/*
		// Сигмоидальная переходная функция нейрона
		// В размерности float имеет смысл вычислять sigmoid только для промежутка от -14 до +15 (При T=0, A=1).
		// Значение сигмоиды при этом проходит от 0 до 1.
		// При аргументе от -4 до 5 сигмоида пробегает от 0.01(7) до 0.99(3).
		// При аргументе от -7 до 7 сигмоида пробегает от 0.000911 до 0.9990889.
		// В формуле ( 1 / (1 + expf(T - A * zz)) )    Величина T влияет на смещение результатов сигмоиды по абсциссе. Величина A "сплющивает" или "растягиавет" сигмоиду.
	*/
	return (1.0 / (1.0 + math.Exp(-1.0*f)))
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
	weight []float64        // веса. Размерность та же
	out    float64          // кэш выходного значения
	outs   []chan<- float64 // Массив каналов. По всем ним передается значение out для разных получателей.
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
	//if delta > 0.0001 { // Если значение изменилось не больше, чем на 0.01%, то сигнал не подаем.
	//if delta > 0.001 { // Если значение изменилось не больше, чем на 0.1%, то сигнал не подаем.
	//if delta > 0.01 { // Если значение изменилось не больше, чем на 1%, то сигнал не подаем.
	if delta > 0.0001 { // Если значение изменилось не больше, чем на 0.01%, то сигнал не подаем.
		for _, c := range N.outs {
			c <- val
		}
		//FIXME
		fmt.Println("Neuron:", N, "returns value:", val)
	}
}

func (N *Neuron) listen() {
	for i, syn := range N.ins {
		go func(s <-chan float64) {
			for {
				select {
				case N.in[i] = <-s:
					N.calc()
				}
			}
		}(syn)
	}
}

func main() {
	// Building NN
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

	n1.link_with(&n2, 0.9)
	n2.link_with(&n1, 0.666)
	// End of building NN

	// Запуск
	n1.outs[0] <- 1.0

	// Работа
	n1.listen()
	n2.listen()
	for {
		//n1.listen()
		//n2.listen()
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("Enter text: ")
		text, _ := reader.ReadString('\n')
		if text[0] == byte("q"[0]) {
			return
		}
		in_int, err := strconv.ParseFloat(strings.Split(text, "\n")[0], 64)
		if err == nil {
			fmt.Println("in_int:", in_int)
			n1.outs[0] <- in_int
		}
	}
}
