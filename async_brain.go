package main

// Online POC version
// https://play.golang.org/p/lA0yztEaOP

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync" // Mutex
	//"time"
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
	ins     []<-chan float64 // Массив входных каналов нейрона (здесь только указатели на них. Создаёт их выдающий сигнал нейрон).
	in      []float64        // Кэш значений входных синапсов
	inmux   []sync.Mutex     // Мьютексы для работы с входными значениями в in.
	weight  []float64        // веса. Размерность та же
	pre_out float64
	out     float64          // кэш выходного значения
	outmux  sync.Mutex       // Мьютекс для работы с выходным значением в out.
	outs    []chan<- float64 // Массив каналов. По всем ним передается значение out для разных получателей.
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
	var m sync.Mutex
	N.inmux = append(N.inmux, m)
	N.weight = append(N.weight, W)
}

func (N *Neuron) calc() {
	val := 0.0
	delta := 0.0     // дельта между текущим значением нейрона и предыдущим.
	pre_delta := 0.0 // дельта между текущим значением нейрона и пред-предыдущим (pre_out).
	for i, _ := range N.inmux {
		(&(N.inmux[i])).Lock() // При включении этого мьютекса у нас идет пересечение с уже включенными мьютексами в listen().
		val += N.in[i] * N.weight[i]
		(&(N.inmux[i])).Unlock()
	}
	val = _sigmoid_(val)
	N.outmux.Lock()
	delta = math.Abs((N.out - val) / N.out)
	pre_delta = math.Abs((N.pre_out - val) / N.pre_out)
	//N.out = val // И не сохраняем новое значение val в N.out.
	N.outmux.Unlock()
	//if (delta > 0.01) && (pre_delta > 0.01) { // Если значение изменилось не больше, чем на 1%, то сигнал не подаем.
	if (delta > 0.001) && (pre_delta > 0.001) { // Если значение изменилось не больше, чем на 0.1%, то сигнал не подаем.
		N.outmux.Lock()   //
		N.pre_out = N.out //
		N.out = val       // И не сохраняем новое значение val в N.out.
		N.outmux.Unlock() // Таким образом, мы даем "накопиться" дельте в несколько этапов, пока меняются значения входных синапсов.

		//FIXME
		//fmt.Println(N.out, ">>", N.outs)
		for _, c := range N.outs {
			fmt.Println("\t\t\t\t\t\t\t\tSending", val, "to", c, "of [", N.outs, "]")
			go func(cc chan<- float64, value float64) {
				//fmt.Println("Sending", value, "into", cc)
				cc <- value
			}(c, val)
			//c <- val
		}
	} else {
		N.outmux.Lock()
		fmt.Println("!!!!!!!!!!!!!!! delta is too low.", val, "(", N.out, ")", "wouldn't be sent to", N.outs)
		N.outmux.Unlock()
	}
}

func (N *Neuron) listen() {
	//N.inmux.Lock()
	//defer N.inmux.Unlock()
	for i, syn := range N.ins {
		go func(NN *Neuron, n int, s <-chan float64) { //FIXME Поведение отличается в зависимости от того 'go func' или просто 'func'. Видимо, надо разбираться с замыканиями.
			// Выяснили, что внутри этой функции использовать 'i' и 'syn' нельзя, а можно только переданные их значения 'n' и 's'.
			//fmt.Println("###", NN, "START LISTENING [", n, "]", s) // видим, что i и n различаются.
			var val float64
			for {
				//fmt.Println(NN.ins, "[", n, "]", ">>to>>", NN.outs) // !!!!!!!!!!!!
				select {
				case val = <-s:
					N.inmux[n].Lock()
					NN.in[n] = val
					N.inmux[n].Unlock()
					//FIXME
					//if debug == 1 {
					fmt.Println("chanel", s, "gotcha in select. n =", n, "(of N.ins", NN.ins, "). Value received:", NN.in[n])
					//}
					go NN.calc()
				}
			}
		}(N, i, syn)
	}
}

func nn_random_constructor(n_in, n_int, n_out, max_syn int) []Neuron {
	var n_neur int = n_in + n_int + n_out
	r := rand.New(rand.NewSource(111237))
	N := make([]Neuron, n_in+n_int+n_out)
	for i, _ := range N[n_in:] { // для всех связанных нейронов
		n := &N[n_in+i]
		for i := 0; i <= r.Intn(max_syn); i++ { // Создаем до max_syn рэндомных синапсов.
			//n.link_with(&N[r.Intn(n_neur)], float64(r.Intn(50))/float64(r.Intn(50)+1.0))
			n.link_with(&N[r.Intn(n_neur)], -3.0+r.Float64()*6.0)
		}
	}
	return N
}

func main() {
	var n_in = 3
	var n_int = 3
	var n_out = 3
	var max_syn = 2
	var N []Neuron = nn_random_constructor(n_in, n_int, n_out, max_syn)
	In := N[:n_in]
	Int := N[n_in : n_in+n_int]
	Out := N[n_in+n_int:]
	Linked := N[n_in:]

	// Работа.
	for i, _ := range Linked { // !!! Если делать 'for i,n := range Linked', то в n будет копия.
		(&Linked[i]).listen()
	}

	// Запуск
	debug = 1
	In[0].outs[0] <- 1.0

	for {
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("Enter text: ")
		text, _ := reader.ReadString('\n')
		if text[0] == "q"[0] {
			return
		}
		if text[0] == "p"[0] {
			fmt.Printf("In:\t%p:\t%+v\n", In, In)
			fmt.Printf("Int:\t%p:\t%+v\n", Int, Int)
			fmt.Printf("Out:\t%p:\t%+v\n\n", Out, Out)
		}
		in_int, err := strconv.ParseFloat(strings.Split(text, "\n")[0], 64)
		if err == nil {
			fmt.Println("_sigmoid_(", in_int, "):", _sigmoid_(in_int))
			//FIXME
			//if in_int == -1 {
			debug = 1
			//}
			In[0].outs[0] <- in_int
		}
	}
}
