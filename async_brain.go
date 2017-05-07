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

type NeurNet struct {
	n_in    int
	n_int   int
	n_out   int
	max_syn int
	n_neur  int
	Neur    []Neuron
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

/* ---=== Mutation ===--- */
func (N *Neuron) weight_change_random() {

}

func (N *Neuron) synapse_add_random() {

}

/*
type Neuron struct {
	in_ch   chan signal             // Единый входной канал
	in      map[*Neuron]float64     // Кэш входных значений
	weight  map[*Neuron]float64     // Веса по указателям источников.
	outs    map[*Neuron]chan signal // Выходные каналы
	out     float64
	pre_out float64
}*/

// Sinapse_Remove
// У нейрона N удаляем синапс, берущий сигнал у нейрона N2 (по указателю).
func (N *Neuron) synapse_del_random(N2 *Neuron) {
	N2.receiver_del(N)
	delete(N.in, N2)
	delete(N.weight, N2)
	if len(N.in) == 0 {
		N.in_ch <- signal{nil, 31337}
	}
	//FIXME Если удаляется последний входящий синапс, то мы должны погасить N.listen()
	// Закрыть канал N.in_ch и освободить память от него и от всего нейрона.
	// Память освобождает garbageCollector, так что, достаточно перестать использовать элемент (по идее, надо перестать на него ссылаться, но это надо уточнить).
	// Чтобы погасить горутину N.listen() можно посылать какое-то спецзначение (чтобы не создавать отдельный управляющий сигнал.).
}

// У нейрона N удаляем передающую часть синапса к N2.in_ch.
// Т.е. удаляется N.outs[targ_N]
func (N *Neuron) receiver_del(N2 *Neuron) {
	delete(N.outs, N2)
}

/*

func (NN Neuron[]) neuron_add_random() {

}

// Neur_Remove
func (NN Neuron[]) neuron_del_random() {

}

func (NN Neuron[]) dump(finlename str) {

}

func (NN Neuron[]) load(filename str) {

}
*/
/* First is blocking select, which gets one signal.
   Then goes unblocking select, which gets other signals if they are already in queue.
   If queue is empty (select default), then we do calc().
*/

// Neur_Remove
/*
func (NN Neuron[]) neuron_del(N *Neuron) {
	fmt.Println("Neuron", N, "would be stopped!")
	close(N.in_ch)
	// Если в N.in что-то есть, перебрать и поотрвать синапсы (synapse_del())
	N.in = nil
	N.weight = nil
	N.outs = nil
	// Должен также быть пересчет слайса нейронов в NN
	// (Перестраиваем со сдвигом всех последующих нейронов на 1 влево.
	//  Затем уменьшаем слайс, отрезая последний элемент)
}
*/
/*
//FIXME
Это просто тест.
Не понятно, как сделать метод для слайса. Судя по всему, это нельзя сделать.
Видимо, надо будет передавать нейросеть параметром.
func (NN *(Neuron[])) neuron_del() {

}
*/

func (N *Neuron) listen() {
	for {
		select {
		case sig := <-N.in_ch:
			if sig.val == 31337 && sig.source == nil {
				//FIXME // Заменить на neuron_del()
				fmt.Println("Neuron", N, "would be stopped!")
				close(N.in_ch)
				N.in = nil
				N.weight = nil
				N.outs = nil
				return
			}
			N.in[sig.source] = sig.val
		}
		select {
		case sig := <-N.in_ch:
			if sig.val == 31337 && sig.source == nil {
				//FIXME // Заменить на neuron_del()
				fmt.Println("Neuron", N, "would be stopped!")
				close(N.in_ch)
				N.in = nil
				N.weight = nil
				N.outs = nil
				return
			}
			//fmt.Println("!!! Unblocked read !!! It's wonderfull !!!", sig) // На дополнительном неблокирующем чтении мы экономим лишние вызовы calc().
			N.in[sig.source] = sig.val
		default:
			N.calc()
		}
	}
}

func nn_random_constructor(n_in, n_int, n_out, max_syn int) []Neuron {
	var NN NeurNet
	NN.n_in = n_in
	NN.n_int = n_int
	NN.n_out = n_out
	NN.max_syn = max_syn
	NN.n_neur = n_in + n_int + n_out
	r := rand.New(rand.NewSource(111237))
	NN := make([]Neuron, n_in+n_int+n_out)
	for i, _ := range NN {
		n := &NN[i]
		// Для входных нейронов это не нужно.
		if i >= n_in {
			n.in_ch = make(chan signal, max_syn)    // Один входной канал для всех синапсов емкостью max_syn.
			n.in = make(map[*Neuron]float64, 1)     // Кэш входных сигналов по указателю отправителя.
			n.weight = make(map[*Neuron]float64, 1) // Карта весов по указателю отправителя.
		}
		n.outs = make(map[*Neuron]chan signal, 10) // Выходные сигналы
	}
	for i, _ := range NN {
		n := &NN[i]
		if i >= n_in {
			for j := 0; j <= r.Intn(max_syn); j++ { // Создаем до max_syn рэндомных синапсов.
				//n.link_with(&NN[r.Intn(n_neur)], float64(r.Intn(50))/float64(r.Intn(50)+1.0))
				n.link_with(&NN[r.Intn(n_neur)], -3.0+r.Float64()*6.0)
			}
		}
	}
	return NN
}

func main() {
	var n_in = 30
	var n_int = 30
	var n_out = 30
	var max_syn = 7
	var NN []Neuron = nn_random_constructor(n_in, n_int, n_out, max_syn)
	In := NN[:n_in]
	Int := NN[n_in : n_in+n_int]
	Out := NN[n_in+n_int:]
	Linked := NN[n_in:]

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
				if in_int == 31337 {
					c <- signal{nil, in_int}
				} else {
					c <- signal{n0, in_int}
				}
				break
			}
		}
	}
}
