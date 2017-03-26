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
	//"sync" // Mutex
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

/* По синапсам передается структура signal.
     source - указатель на нейон отправитель.
	 val - значение сигнала.
*/
type signal struct {
	source *Neuron
	val    float64
}

type Neuron struct {
	in_ch   chan signal             // Единый входной канал
	in      map[*Neuron]float64     // Кэш входных значений
	weight  map[*Neuron]float64     // Веса по указателям источников.
	outs    map[*Neuron]chan signal // Выходные каналы
	out     float64
	pre_out float64
}

// Связать нейрон N c нейроном N2 с весом weight
// N2 будет испускать сигналы, а N будет их обрабатывать.
func (N *Neuron) link_with(N2 *Neuron, weight float64) {
	N.weight[N2] = weight
	N2.outs[N] = N.in_ch
}

func (N *Neuron) calc() {
	val := 0.0
	delta := 0.0     // дельта между текущим значением нейрона и предыдущим.
	pre_delta := 0.0 // дельта между текущим значением нейрона и пред-предыдущим (pre_out).
	for n, v := range N.in {
		val += v * N.weight[n]
	}
	val = _sigmoid_(val)

	delta = math.Abs((N.out - val) / N.out)
	pre_delta = math.Abs((N.pre_out - val) / N.pre_out)
	//if (delta > 0.01) && (pre_delta > 0.01) { // Если значение изменилось не больше, чем на 1%, то сигнал не подаем.
	if delta > 0.001 && (pre_delta > 0.001) { // Если значение изменилось не больше, чем на 0.1%, то сигнал не подаем.
		N.pre_out = N.out // И не сохраняем новое значение val в N.out.
		N.out = val       // Таким образом, мы даем "накопиться" дельте в несколько этапов, пока меняются значения входных синапсов.
		//FIXME
		//fmt.Println(N.out, ">>", N.outs)
		for _, c := range N.outs {
			//FIXME
			fmt.Println("\t\t\t\tSending", val, "to", c, "of [", N.outs, "]")
			go func(cc chan<- signal, value float64) {
				//FIXME
				//fmt.Println("Sending", value, "into", cc)
				cc <- signal{N, value}
			}(c, val)
			//c <- val
		}
	} else {
		//FIXME
		//fmt.Println("!!!!!!!!!!!!!!! delta is too low.", val, "(", N.out, ")", "wouldn't be sent to", N.outs)
	}

}

/* First is blocking select, which gets one signal.
   Then goes unblocking select, which gets other signals if they are already in queue.
   If queue is empty (select default), then we do calc().
*/
func (N *Neuron) listen() {
	for {
		select {
		case sig := <-N.in_ch:
			N.in[sig.source] = sig.val
		}
		select {
		case sig := <-N.in_ch:
			fmt.Println("!!! Unblocked read !!! It's wonderfull !!!", sig) // На дополнительном неблокирующем чтении мы экономим лишние вызовы calc().
			N.in[sig.source] = sig.val
		default:
			N.calc()
		}
	}
}

func nn_random_constructor(n_in, n_int, n_out, max_syn int) []Neuron {
	var n_neur int = n_in + n_int + n_out
	r := rand.New(rand.NewSource(111237))
	N := make([]Neuron, n_in+n_int+n_out)
	for i, _ := range N {
		n := &N[i]
		// Для входных нейронов это не нужно.
		if i >= n_in {
			n.in_ch = make(chan signal, max_syn)    // Один входной канал для всех синапсов емкостью max_syn.
			n.in = make(map[*Neuron]float64, 1)     // Кэш входных сигналов по указателю отправителя.
			n.weight = make(map[*Neuron]float64, 1) // Карта весов по указателю отправителя.
		}
		n.outs = make(map[*Neuron]chan signal, 10) // Выходные сигналы
	}
	for i, _ := range N {
		n := &N[i]
		if i >= n_in {
			for j := 0; j <= r.Intn(max_syn); j++ { // Создаем до max_syn рэндомных синапсов.
				//n.link_with(&N[r.Intn(n_neur)], float64(r.Intn(50))/float64(r.Intn(50)+1.0))
				n.link_with(&N[r.Intn(n_neur)], -3.0+r.Float64()*6.0)
			}
		}
	}
	return N
}

func main() {
	var n_in = 30
	var n_int = 30
	var n_out = 30
	var max_syn = 7
	var N []Neuron = nn_random_constructor(n_in, n_int, n_out, max_syn)
	In := N[:n_in]
	Int := N[n_in : n_in+n_int]
	Out := N[n_in+n_int:]
	Linked := N[n_in:]

	// Работа.
	for i, _ := range Linked { // !!! Если делать 'for i,n := range Linked', то в n будет копия нейрона.
		go (&Linked[i]).listen()
	}

	// Запуск
	debug = 1
	n0 := &In[0]
	for _, c := range n0.outs {
		c <- signal{n0, 1.0}
		break // отправляем только в самый первый нейрон
	}

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
			for _, c := range n0.outs { // 'n0' остался из раздела "Запуск"
				c <- signal{n0, in_int}
				break
			}
		}
	}
}
