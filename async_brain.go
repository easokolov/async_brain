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

//var debug int32 = 1

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
	out     float64                 // Последнее выходное значение.
	pre_out float64                 // Предыдущее выходное значение.
}

type NeurNet struct {
	n_in    int
	n_int   int
	n_out   int
	max_syn int
	n_neur  int
	Neur    []Neuron
	// In, Int, Out, Linked - поверх того же массива, на который указывает Neur.
	// ! Если в Neur добвить нейрон, то массив будет переразмещен, и все эти слайсы надо будет переопределить !
	// ! Иначе они продолжат указывать на старый массив !
	In     []Neuron
	Int    []Neuron
	Out    []Neuron
	Linked []Neuron
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
	if len(N.weight) == 0 {
		// Это входной нейрон. Входной канал есть, и на него может быть подано значение.
		// Но оно не будет связано с каким-то другим нейроном. Веса нет. Приходит от nil.
		// Это же значение будем передавать на выход без каких-либо преобразований.
		val = N.in[nil]
	} else {
		for n, v := range N.in {
			val += v * N.weight[n]
		}
		val = _sigmoid_(val)
	}

	delta = math.Abs((N.out - val) / N.out)
	pre_delta = math.Abs((N.pre_out - val) / N.pre_out)
	//if (delta > 0.01) && (pre_delta > 0.01) { // Если значение изменилось не больше, чем на 1%, то сигнал не подаем.
	if (delta > 0.001 && (pre_delta > 0.001)) || len(N.weight) == 0 { // Если значение изменилось не больше, чем на 0.1%, то сигнал не подаем.
		// Кроме входных нейронов. Если на них приходит старое значение, все равно передаем в НС.
		N.pre_out = N.out // И не сохраняем новое значение val в N.out.
		N.out = val       // Таким образом, мы даем "накопиться" дельте в несколько этапов, пока меняются значения входных синапсов.
		for _, c := range N.outs {
			go func(cc chan<- signal, value float64) {
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
//func (N *Neuron) synapse_del_random(N2 *Neuron) {
func (N *Neuron) synapse_del(N2 *Neuron) {
	delete(N2.outs, N)
	delete(N.in, N2)
	delete(N.weight, N2)
	//if len(N.in) == 0 {
	//	N.in_ch <- signal{nil, 31337}
	//}
	// Дернуть neuron_del() должен вызывающий synapse_del()
	//FIXME Если удаляется последний входящий синапс, то мы должны погасить N.listen().
	// Закрыть канал N.in_ch и освободить память от него и от всего нейрона.
	// Чтобы погасить горутину N.listen() можно посылать какое-то спецзначение (чтобы не создавать отдельный управляющий сигнал.).
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
func (NN *NeurNet) neuron_del(N *Neuron) {
	fmt.Println("Neuron", N, "would be stopped!")
	// If there is something in N.in or N.outs, we should itterete it and remove synapses.
	for n, _ := range N.weight {
		N.synapse_del(n)
	}
	for n, _ := range N.outs {
		n.synapse_del(N)
		//if len(n.in) == 0 {
		//	NN.neuron_del(n)
		//}
	}
	N.in = nil
	N.weight = nil
	N.outs = nil
	// Должен также быть пересчет слайса нейронов в NN.Neur
	// (Перестраиваем со сдвигом всех последующих нейронов на 1 влево.
	//  Затем уменьшаем слайс, отрезая последний элемент)

	//Если нейрон входной, то у него не создан канал. Надо либо создавать канал, либо проверять канал на !=nil и не делать отправку 31337!
	if N.in_ch != nil {
		N.in_ch <- signal{nil, 31337}
		close(N.in_ch)
	}
	for i, _ := range NN.Neur {
		if &(NN.Neur[i]) == N {
			fmt.Printf("!!!!!!!!!!!!We found a deleted neuron with index %v. We should delete it\n", i)
		}
	}
}

// Couple of selects (blocked and unblocked) are for reducing calc()-s.
// (Will receive all ready incomings in incoming chanels befor calling calc()).
// receive() just doing every select must to do (for reducing the code).
func (NN *NeurNet) listen(N *Neuron) {
	// if receive() returns false, then listen() should be exit too.
	receive := func(N *Neuron, sig signal) bool {
		if sig.val == 31337 && sig.source == nil {
			//NN.neuron_del(N)
			return (false)
		}
		N.in[sig.source] = sig.val
		return (true)
	}

	for {
		select {
		case sig := <-N.in_ch:
			if !receive(N, sig) {
				return
			}
		}
		select {
		case sig := <-N.in_ch:
			if !receive(N, sig) {
				return
			}
			//fmt.Println("!!! Unblocked read !!! It's wonderfull !!!", sig) // На дополнительном неблокирующем чтении мы экономим лишние вызовы calc().
		default:
			N.calc()
		}
	}
}

func nn_random_constructor(n_in, n_int, n_out, max_syn int) NeurNet {
	var NN NeurNet
	NN.n_in = n_in
	NN.n_int = n_int
	NN.n_out = n_out
	NN.max_syn = max_syn
	NN.n_neur = n_in + n_int + n_out
	NN.Neur = make([]Neuron, NN.n_neur)
	NN.In = NN.Neur[:n_in]
	NN.Int = NN.Neur[n_in : n_in+n_int]
	NN.Out = NN.Neur[n_in+n_int:]
	NN.Linked = NN.Neur[n_in:]
	r := rand.New(rand.NewSource(111237))
	for i, _ := range NN.Neur {
		n := &(NN.Neur[i])
		n.in_ch = make(chan signal, max_syn)      // Один входной канал для всех синапсов емкостью max_syn.
		n.outs = make(map[*Neuron]chan signal, 1) // Выходные сигналы. Можно задать начальную емкость.
		n.in = make(map[*Neuron]float64, 1)       // Кэш входных сигналов по указателю отправителя. (У входных нейронов указатель отправителя - всегда nil.)
		// Для входных нейронов это не нужно.
		if i >= n_in {
			n.weight = make(map[*Neuron]float64, 1) // Карта весов по указателю отправителя.
		}
	}
	for i, _ := range NN.Neur {
		n := &(NN.Neur[i])
		if i >= n_in { // Not for incoming neurons.
			for j := 0; j <= r.Intn(max_syn); j++ { // Создаем до max_syn рэндомных синапсов.
				//n.link_with(&NN[r.Intn(NN.n_neur)], float64(r.Intn(50))/float64(r.Intn(50)+1.0))
				n.link_with(&(NN.Neur[r.Intn(NN.n_neur)]), -3.0+r.Float64()*6.0)
			}
		}
	}
	return NN
}

//FIXME
func out(s string) {
	fmt.Println(s)
}

func main() {
	var NN NeurNet = nn_random_constructor(30, 30, 30, 7)

	// Работа.
	for i, _ := range NN.Linked { // !!! Если делать 'for i,n := range NN.Linked', то в n будет копия нейрона.
		go (&NN).listen(&(NN.Linked[i]))
	}

	// Входные нейроны теперь тоже имеют входной канал.
	for i, _ := range NN.In {
		go (&NN).listen(&(NN.In[i]))
	}

	// Запуск
	n0 := &(NN.In[0])
	for _, c := range n0.outs {
		c <- signal{n0, 1.0}
		break // отправляем только в самый первый нейрон
	}

	// Читаем клаву. Ждем команд или входов в n0
	for {
		out("000")
		n0 = &(NN.In[0])
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("Enter command or input for n0: ")
		text, _ := reader.ReadString('\n')
		if text[0] == "q"[0] {
			return
		}
		if text[0] == "p"[0] {
			fmt.Printf("In:\t%p:\t%+v\n", NN.In, NN.In)
			fmt.Printf("Int:\t%p:\t%+v\n", NN.Int, NN.Int)
			fmt.Printf("Out:\t%p:\t%+v\n\n", NN.Out, NN.Out)
		}
		if text[0] == "e"[0] {
			fmt.Println("n0:", n0)
		}
		input, err := strconv.ParseFloat(strings.Split(text, "\n")[0], 64)
		if err == nil {
			fmt.Println("_sigmoid_(", input, "):", _sigmoid_(input))
			out("a00")
			fmt.Println("n0.outs =", n0.outs)

			out("a01")
			if input == 31337 {
				out("a02")
				(&NN).neuron_del(n0)
				out("a03")
				continue
			}

			n0.in_ch <- signal{nil, input}
			//for _, c := range n0.outs {
			//	c <- signal{n0, input}
			//}
		}
	}
}
