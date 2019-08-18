package async_brain

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
	"encoding/json"
	"runtime"
	"time"
)

//var debug int32 = 1
var r *rand.Rand = rand.New(rand.NewSource(111237))

//var r *rand.Rand = rand.New(rand.NewSource(111236))

func round3(f float64) float64 {
	return math.Trunc(f*1000) / 1000
}

func remove(slice []*Neuron, i int) []*Neuron {
	copy(slice[i:], slice[i+1:])
	return slice[:len(slice)-1]
}

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

/* По синапсам передается структура Signal.
     source - указатель на нейон отправитель.
	 val - значение сигнала.
*/
type Signal struct {
	source *Neuron
	val    float64
}

type Neuron struct {
	in_ch   chan Signal             // Единый входной канал
	in      map[*Neuron]float64     // Кэш входных значений
	weight  map[*Neuron]float64     // Веса по указателям источников.
	outs    map[*Neuron]chan Signal // Выходные каналы
	out     float64                 // Последнее выходное значение.
	pre_out float64                 // Предыдущее выходное значение.
}

type NeurNet struct {
	n_in     int
	n_int    int
	n_out    int
	max_syn  int
	n_linked int
	n_neur   int
	Neur     []*Neuron
	// In, Int, Out, Linked - поверх того же массива, на который указывает Neur.
	// ! Если в Neur добавить нейрон, то массив будет переразмещен, и все эти слайсы надо будет переопределить !
	// ! Иначе они продолжат указывать на старый массив !
	// ! Если удалить нейрон, но границы групп также изменятся. Также надо переопределять.
	In     []*Neuron
	Int    []*Neuron
	Out    []*Neuron
	Linked []*Neuron
	//mutex   sync.Mutex
}

type NeurIndex int
type DumpedNeuron struct {
	In      map[NeurIndex]float64 // Кэш входных значений
	Weight  map[NeurIndex]float64 // Веса по указателям источников.
	Outs    []NeurIndex           // Выходные каналы
	Out     float64               // Последнее выходное значение.
	Pre_out float64               // Предыдущее выходное значение.
}
type DumpedNeurNet struct {
	N_in     int
	N_int    int
	N_out    int
	Max_syn  int
	N_linked int
	N_neur   int
	Neur     []DumpedNeuron
}

func (NN *NeurNet) Dump(filePath string) error {
	// nmap временная структура для быстрого поиска индексов нейронов в NN.Neur по их указателю.
	nmap := make(map[*Neuron]NeurIndex)
	for i, n := range NN.Neur {
		nmap[n] = NeurIndex(i)
	}
	// Копируем данные из NN в DNN c заменой указателей нейронов на индексы. Дампить будем DNN.
	var DNN DumpedNeurNet
	DNN.N_in = NN.n_in
	DNN.N_int = NN.n_int
	DNN.N_out = NN.n_out
	DNN.Max_syn = NN.max_syn
	DNN.N_linked = NN.n_linked
	DNN.N_neur = NN.n_neur
	DNN.Neur = make([]DumpedNeuron, NN.n_neur)
	for i, n := range NN.Neur {
		// Копаирует NN.Neur[*].in в DNN.Neur[*].in с заменой указателей нейронов на индексы
		DNN.Neur[i].In = make(map[NeurIndex]float64, len(n.in))
		for k, v := range n.in {
			DNN.Neur[i].In[nmap[k]] = v
		}
		// Копаирует NN.Neur[*].weight в DNN.Neur[*].weight с заменой указателей нейронов на индексы
		DNN.Neur[i].Weight = make(map[NeurIndex]float64, len(n.weight))
		for k, v := range n.weight {
			DNN.Neur[i].Weight[nmap[k]] = v
		}
		// Копируем NN.Neur[*].outs в DNN.Neur[*].outs, только теперь это не маз указателя нейрона-получателя и его входного канала а только массив индексов получателей.
		DNN.Neur[i].Outs = make([]NeurIndex, len(n.outs))
		ind := 0
		for k, _ := range n.outs {
			DNN.Neur[i].Outs[ind] = nmap[k]
			ind++
		}
		DNN.Neur[i].Out = n.out
		DNN.Neur[i].Pre_out = n.pre_out
	}

	file, err := os.Create(filePath)
	if err == nil {
		encoder := json.NewEncoder(file)
		encoder.Encode(DNN)
	}
	file.Close()
	return err
}

func (NN *NeurNet) Load(filePath string) error {
	DNN := new(DumpedNeurNet)
	file, err := os.Open(filePath)
	defer file.Close()
	if err != nil {
		return err
	}
	decoder := json.NewDecoder(file)
	err = decoder.Decode(DNN)
	if err != nil {
		return err
	}
	//FIXME debug
	//fmt.Printf("DNN = '%+v'\n", DNN)
	NN.n_in = DNN.N_in
	NN.n_int = DNN.N_int
	NN.n_out = DNN.N_out
	NN.max_syn = DNN.Max_syn
	NN.n_linked = DNN.N_linked
	NN.n_neur = DNN.N_neur

	NN.Neur = make([]*Neuron, 0)
	for i := 0; i < NN.n_neur; i++ {
		NN.Neur = append(NN.Neur, new(Neuron))
		NN.Neur[i].in_ch = make(chan Signal, NN.max_syn)
	}
	NN.set_slices(NN.n_in, NN.n_int, NN.n_out)
	for i, n := range NN.Neur {
		n.outs = make(map[*Neuron]chan Signal, len(DNN.Neur[i].Outs))
		for _, ind := range DNN.Neur[i].Outs {
			n.outs[NN.Neur[ind]] = NN.Neur[ind].in_ch
		}
		n.in = make(map[*Neuron]float64, len(DNN.Neur[i].In))
		for ind, val := range DNN.Neur[i].In {
			n.in[NN.Neur[ind]] = val
		}
		// Для входных нейронов это не нужно.
		if i >= NN.n_in {
			n.weight = make(map[*Neuron]float64, len(DNN.Neur[i].Weight))
			for ind, val := range DNN.Neur[i].Weight {
				n.weight[NN.Neur[ind]] = val
			}
		}
	}
	return err
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
		// Но ононе будет связано с каким-то другим нейроном. Веса нет. Приходит от nil.
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
		fmt.Printf("%p -> %v -> %v\n", N, val, N.outs)
		for _, c := range N.outs {
			go func(c chan<- Signal, val float64) {
				//fmt.Printf("%p_calc()_pre_send to %p; val=%v\n", N, c, val)
				c <- Signal{N, val}
				//fmt.Printf("%p_calc()_post_send to %p; val=%v\n", N, c, val)
			}(c, val)
			//c <- Signal{N, val}
		}
	} else {
		//FIXME
		//fmt.Println("!!!!!!!!!!!!!!! delta is too low.", val, "(", N.out, ")", "wouldn't be sent to", N.outs)
	}
}

/* ---=== Mutation ===--- */
// Произвольно меняем вес произвольного синапса нейрона N на случайную дельту (от -0.05 до 0.05).
// В одном случае из 10 дельта увеличиваеся в диапазоне от -0.5 до 0.5
func (N *Neuron) weight_change_random() {
	index := r.Intn(len(N.weight)) // Выбираем случайный синапс
	i := 0
	for n, _ := range N.weight {
		if i == index {
			multiplier := 0.1
			if r.Intn(10) == 9 {
				multiplier = 1.0
			}
			N.weight[n] += (r.Float64() - 0.5) * multiplier
			break
		}
		i++
	}
}

func (NN *NeurNet) weight_change_random() {
	n := NN.Linked[r.Intn(NN.n_linked)] // Weight will be changed for n (random Linked neuron)
	n.weight_change_random()
}

func (NN *NeurNet) synapse_add_random() {
	n := NN.Linked[r.Intn(NN.n_linked)] // Synapse will be added for n (random Linked neuron)
	if len(n.weight) < NN.max_syn {
		n_target := NN.Neur[r.Intn(NN.n_neur)]
		n.link_with(n_target, -3.0+r.Float64()*6.0)
	} else {
		out(fmt.Sprintf("Neuron synapses limit is exceeded for %p.", n))
	}
}

// Возвращает указатель на нейрон N2, к которому ведет случайный синапс нейрона N.
func (N *Neuron) get_random_synapse() (N2 *Neuron, err error) {
	if len(N.weight) == 0 {
		return nil, fmt.Errorf("get_random_synapse(): Neuron %v doesn't have any synapse", N)
	}
	index := r.Intn(len(N.weight))
	i := 0
	for N2, _ = range N.weight {
		if i == index {
			break
		}
		i++
	}
	return N2, nil
}

// У нейрона N удаляем синапс, берущий сигнал у нейрона N2 (по указателю).
func (N *Neuron) synapse_del(N2 *Neuron) {
	delete(N2.outs, N)
	delete(N.in, N2)
	delete(N.weight, N2)
	// Если удаляется последний входящий синапс, то
	// дернуть neuron_del() должен вызывающий synapse_del()
	// Т.к. neuron_del() может дергаться и самостоятельно,
	// и в этом случае он должен вызывать synapse_del().
}

func (N *Neuron) synapse_del_random() {
	N2, err := N.get_random_synapse()
	if err != nil {
		fmt.Println(err)
		return
	}
	N.synapse_del(N2)
	// Нейрон без связей будет удален в NN.synapse_del_random()
	// т.к. здесь мы не имеем указателя NN
}

func (NN *NeurNet) synapse_del_random() {
	if NN.n_linked == 0 {
		panic("synapse_del_random(): we have no linked neurons.")
	}
	N := NN.Linked[r.Intn(NN.n_linked)] // Synapse will be deleted for N (random Linked neuron)
	N.synapse_del_random()
	// Если это внутренний нейрон и у него не осталось синапсов, то удаляем его (но не выходной)
	if len(N.weight) == 0 {
		if NN.get_index(N) < NN.n_in+NN.n_int {
			NN.neuron_del(N)
		}
	}
}

// /*

func (NN *NeurNet) neuron_add_random() {
	//Должно строиться на основе NN.Neur = append(NN.Neur, new(Neuron))
	//только со смещением Out-нейронов в конец и переопределением слайсов Int, Out, Linked
	NN.Neur = append(NN.Neur, nil)
	// Смещаем out-нейроны в конец (copy(dst, src)).
	NN.set_slices(NN.n_in, NN.n_int+1, NN.n_out)
	ncopy := copy(NN.Neur[NN.n_neur-NN.n_out:NN.n_neur], NN.Neur[NN.n_neur-NN.n_out-1:NN.n_neur-1])
	if ncopy == NN.n_out {
		//out(fmt.Sprintf("!!! OK! NN.n_out (%v) neurons copied.", NN.n_out))
	} else {
		out(fmt.Sprintf("!!! BAD! %v neurons copied. (!= NN.n_out (%v)).", ncopy, NN.n_out))
	}
	// Переопределяем последний int-нейрон (создаем новый)
	newi := NN.n_in + NN.n_int - 1 // index of new internal newron.
	NN.Neur[newi] = new(Neuron)
	NN.Neur[newi].in_ch = make(chan Signal, NN.max_syn)   // Один входной канал для всех синапсов емкостью max_syn.
	NN.Neur[newi].outs = make(map[*Neuron]chan Signal, 1) // Выходные сигналы. Можно задать начальную емкость.
	NN.Neur[newi].in = make(map[*Neuron]float64, 1)       // Кэш входных сигналов по указателю отправителя.
	NN.Neur[newi].weight = make(map[*Neuron]float64, 1)   // Карта весов по указателю отправителя.
	//for j := 0; j <= r.Intn(NN.max_syn); j++ {          // Создаем до max_syn рэндомных синапсов нового нейрона. (кажется, это слишком жирно. Слишком большое изменение)
	for j := 0; j <= r.Intn(2); j++ { // Создаем до 2 рэндомных синапсов нового нейрона. (Пока будем созавать 1-2 синапса, а не до max_syn)
		NN.Neur[newi].link_with(NN.Neur[r.Intn(NN.n_neur)], -3.0+r.Float64()*6.0)
	}
	NN.Linked[r.Intn(NN.n_linked)].link_with(NN.Neur[newi], -3.0+r.Float64()*6.0) // Создаем для рэндомного Linked-нейрона синапс на новый нейрон.
	out(fmt.Sprintf("Random neuron added. %v in-, %v internal-, %v out- neurons now.", NN.n_in, NN.n_int, NN.n_out))
}

// func (NN *NeurNet) dump(finlename str) {
//
// }
//
// func (NN *NeurNet) load(filename str) {
//
// }
// */

// Удалить случайный внутренний нейрон.
func (NN *NeurNet) neuron_del_random() {
	NN.neuron_del(NN.Int[r.Intn(NN.n_int)])
}

// при удалении нейрона надо удалить все синапсы, если они есть, погасить N.listen(),
// Закрыть канал N.in_ch и освободить память от него и от всего нейрона.
// Чтобы погасить горутину N.listen() будем посылать спецзначение 31337
// (чтобы не создавать отдельный управляющий сигнал).
func (NN *NeurNet) neuron_del(N *Neuron) {
	out(fmt.Sprintf("Neuron %p would be stopped!", N))
	// If there is something in N.in or N.outs, we should itterate it and remove synapses.
	for n, _ := range N.weight {
		N.synapse_del(n)
	}
	for n, _ := range N.outs {
		n.synapse_del(N)
		// Если это был единственный синапс у n (некий нейрон, бурещий сигнал у N),
		// то такой нейрон можно удалить. Но не стоит каскадно удалять входные и выходные нейроны.
		if len(n.weight) == 0 {
			ind := NN.get_index(n)
			if ind >= NN.n_in && ind < NN.n_in+NN.n_int {
				NN.neuron_del(n)
			}
		}
	}
	N.in = nil
	N.weight = nil
	N.outs = nil

	// Раньше у входных нейронов не было входного канала и для них не надо было посылать 31337.
	// Сейчас не актуально, но проверка на всякий случай пусть будет.
	if N.in_ch != nil {
		// Sending 31337 into incoming chanel stops the listen thread of neuron.
		N.in_ch <- Signal{nil, 31337}
		time.Sleep(1 * time.Second) // Время на получение обработку сигнала 31337
		close(N.in_ch)
	}

	index := NN.get_index(N)
	if index < 0 {
		out(fmt.Sprintf("Neuron %p not in NN.Neur", N))
		return
	} else if index < NN.n_in {
		NN.Neur = remove(NN.Neur, index)
		NN.set_slices(NN.n_in-1, NN.n_int, NN.n_out)
	} else if index < NN.n_in+NN.n_int {
		NN.Neur = remove(NN.Neur, index)
		NN.set_slices(NN.n_in, NN.n_int-1, NN.n_out)
	} else if index < NN.n_in+NN.n_int+NN.n_out {
		NN.Neur = remove(NN.Neur, index)
		NN.set_slices(NN.n_in, NN.n_int, NN.n_out-1)
	} else {
		out(fmt.Sprintf("index %v of neuron %p out of NN.Neur!!! Imposible!!!", index, N))
		return
	}
}

// Returns index of N in NN.Neur or -1 if not found.
func (NN *NeurNet) get_index(N *Neuron) int {
	for i, n := range NN.Neur {
		if N == n {
			return (i)
		}
	}
	return (-1)
}

// First is blocking select, which gets one Signal.
// Then goes unblocking select, which gets other Signals if they are already in queue.
// If queue is empty (select default), then we do calc().
// receive() just doing every select must to do (for reducing the code).
func (NN *NeurNet) listen(N *Neuron) {
	defer out(fmt.Sprintf("%p.listen() closed!", N))
	// if receive() returns false, then listen() should be exit too.
	receive := func(N *Neuron, sig Signal) bool {
		if sig.val == 31337 && sig.source == nil {
			return (false) // Выход из listen()
		}
		N.in[sig.source] = sig.val
		return (true) // Продолжить listen()
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
			// На дополнительном неблокирующем чтении мы экономим лишние вызовы calc().
			//fmt.Println("!!! Unblocked read !!! It's wonderfull !!!", sig)
		default:
		}
		N.calc()
	}
}

func (NN *NeurNet) set_slices(n_in, n_int, n_out int) {
	NN.n_in = n_in
	NN.n_int = n_int
	NN.n_out = n_out
	NN.n_linked = n_int + n_out
	NN.n_neur = n_in + n_int + n_out
	NN.In = NN.Neur[:n_in]
	NN.Int = NN.Neur[n_in : n_in+n_int]
	NN.Out = NN.Neur[n_in+n_int:]
	NN.Linked = NN.Neur[n_in:]
}

func nn_random_constructor(n_in, n_int, n_out, max_syn int) *NeurNet {
	var NN NeurNet
	NN.max_syn = max_syn
	NN.n_neur = n_in + n_int + n_out
	NN.Neur = make([]*Neuron, 0)
	for i := 0; i < NN.n_neur; i++ {
		NN.Neur = append(NN.Neur, new(Neuron))
	}
	(&NN).set_slices(n_in, n_int, n_out)
	for i, n := range NN.Neur {
		n.in_ch = make(chan Signal, max_syn)      // Один входной канал для всех синапсов емкостью max_syn.
		n.outs = make(map[*Neuron]chan Signal, 1) // Выходные сигналы. Можно задать начальную емкость.
		n.in = make(map[*Neuron]float64, 1)       // Кэш входных сигналов по указателю отправителя. (У входных нейронов указатель отправителя - всегда nil.)
		// Для входных нейронов это не нужно.
		if i >= n_in {
			n.weight = make(map[*Neuron]float64, 1) // Карта весов по указателю отправителя.
		}
	}
	for _, n := range NN.Linked {
		for j := 0; j <= r.Intn(max_syn); j++ { // Создаем до max_syn рэндомных синапсов.
			n.link_with(NN.Neur[r.Intn(NN.n_neur)], -3.0+r.Float64()*6.0)
		}
	}
	return &NN
}

//FIXME
// Hook for logging purposes
func out(s string) {
	fmt.Println(s)
}

func (NN *NeurNet) Print() {
	fmt.Printf("%+v\n\n", NN)
	for i, n := range NN.In {
		fmt.Printf("NN.In[%v]  = %+v\n", i, n)
	}
	for i, n := range NN.Int {
		fmt.Printf("NN.Int[%v]  = %+v\n", i, n)
	}
	for i, n := range NN.Out {
		fmt.Printf("NN.Out[%v]  = %+v\n", i, n)
	}
}

func (NN *NeurNet) Print_weights() {
	fmt.Printf("In:\t%p:\t%+v\n", NN.In, NN.In)
	fmt.Printf("Int:\t%p:\t%+v\n", NN.Int, NN.Int)
	for _, n := range NN.Int {
		var w []float64
		for _, ww := range n.weight {
			w = append(w, round3(ww))
		}
		fmt.Printf("%v\t", w)
	}
	fmt.Printf("\n")
	fmt.Printf("Out:\t%p:\t%+v\n", NN.Out, NN.Out)
	for _, n := range NN.Out {
		var w []float64
		for _, ww := range n.weight {
			w = append(w, round3(ww))
		}
		fmt.Printf("%v\t", w)
	}
	fmt.Printf("\n\n")
}

/*
func main() {
	var NN *NeurNet = nn_random_constructor(1, 3, 1, 3)
	//NN.Print()

	// !!! Пока NN.Neur был массив структур, а не массив указателей, то делая 'for _,n := range NN.Linked', в n была копия нейрона.
	// Работа.
	for _, n := range NN.Linked {
		go NN.listen(n)
	}

	// Входные нейроны теперь тоже имеют входной канал.
	for _, n := range NN.In {
		go NN.listen(n)
	}

	// Запуск
	n0 := NN.In[0]
	for _, c := range n0.outs {
		c <- Signal{n0, 1.0}
		break // отправляем только в самый первый нейрон
	}

	// Читаем клаву. Ждем команд или входов в n0
	for {
		//out("000")
		n0 = NN.In[0]
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("Enter command or input for n0: ")
		text, _ := reader.ReadString('\n')
		if text[0] == 'q' {
			return
		}
		if text[0] == 'p' {
			if text[1] == 'p' {
				NN.Print()
			} else {
				NN.Print_weights()
			}
			continue
		}
		if text[0] == 'e' {
			fmt.Printf("n0:\t%+v\n", n0)
			continue
		}
		input, err := strconv.ParseFloat(strings.Split(text, "\n")[0], 64)
		if err == nil {
			if input == 31337 {
				NN.neuron_del_random()
				continue
			}
			if input == 31338 {
				NN.synapse_add_random()
				continue
			}
			if input == 31339 {
				NN.weight_change_random()
				continue
			}
			if input == 31340 {
				NN.synapse_del_random()
				//(&NN).Int[0].synapse_del(NN.In[0])
				//(&NN).Int[0].synapse_del(NN.Int[0])
				continue
			}
			if input == 31341 {
				NN.neuron_add_random()
				continue
			}
			if input == 31342 {
				NN.Dump("NN_dump.json")
				continue
			}
			if input == 31343 {
				if NN != nil {
					for _, n := range NN.Neur {
						n.in_ch <- Signal{nil, 31337}
						time.Sleep(1 * time.Second) // Время на получение обработку сигнала 31337
						close(n.in_ch)
					}
				}
				NN = new(NeurNet)
				NN.Load("1")
				for _, n := range NN.Neur {
					go NN.listen(n)
				}
				continue
			}
			if input == 31344 {
				if NN != nil {
					for _, n := range NN.Neur {
						n.in_ch <- Signal{nil, 31337}
						time.Sleep(1 * time.Second) // Время на получение обработку сигнала 31337
						close(n.in_ch)
					}
				}
				NN = new(NeurNet)
				NN.Load("2")
				for _, n := range NN.Neur {
					go NN.listen(n)
				}
				continue
			}
			if input == 31345 {
				runtime.GC() // Запустить GarbageCollector.
				continue
			}

			n0.in_ch <- Signal{nil, input}
			//NN.In[1].in_ch <- Signal{nil, input}
			//for _, c := range n0.outs {
			//	c <- Signal{n0, input}
			//}
		}
	}
}
*/
