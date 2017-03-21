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
	source *NeuronX
	val    float64
}

type NeuronX struct {
	in_ch   chan signal              // Единый входной канал
	in      map[*NeuronX]float64     // Кэш входных значений
	weight  map[*NeuronX]float64     // Веса по указателям источников.
	outs    map[*NeuronX]chan signal // Выходные каналы
	out     float64
	pre_out float64
}

// Связать нейрон N c нейроном N2 с весом weight
// N2 будет испускать сигналы, а N будет их обрабатывать.
func (N *NeuronX) link_withX(N2 *NeuronX, weight float64) {
	N.weight[N2] = weight
	N2.outs[N] = N.in_ch
}

func (N *NeuronX) calc() {
	val := 0.0
	delta := 0.0     // дельта между текущим значением нейрона и предыдущим.
	pre_delta := 0.0 // дельта между текущим значением нейрона и пред-предыдущим (pre_out).
	for n, v := range N.in {
		val += v * N.weight[n]
	}
	val = _sigmoid_(val)
	fmt.Println("In da calc(): sigmoid=", val)

	delta = math.Abs((N.out - val) / N.out)
	pre_delta = math.Abs((N.pre_out - val) / N.pre_out)
	//if (delta > 0.01) && (pre_delta > 0.01) { // Если значение изменилось не больше, чем на 1%, то сигнал не подаем.
	if (delta > 0.001) && (pre_delta > 0.001) { // Если значение изменилось не больше, чем на 0.1%, то сигнал не подаем.
		N.pre_out = N.out // И не сохраняем новое значение val в N.out.
		N.out = val       // Таким образом, мы даем "накопиться" дельте в несколько этапов, пока меняются значения входных синапсов.
		//FIXME
		//fmt.Println(N.out, ">>", N.outs)
		for _, c := range N.outs {
			//FIXME
			fmt.Println("\t\t\t\t\t\t\t\tSending", val, "to", c, "of [", N.outs, "]")
			go func(cc chan<- signal, value float64) {
				//FIXME
				//fmt.Println("Sending", value, "into", cc)
				cc <- signal{N, value}
			}(c, val)
			//c <- val
		}
	} else {
		//FIXME
		fmt.Println("!!!!!!!!!!!!!!! delta is too low.", val, "(", N.out, ")", "wouldn't be sent to", N.outs)
	}

}

func (N *NeuronX) listen() {
	for {
		for i := range N.in_ch {
			//FIXME
			fmt.Println("Neuron", N, "received i=", i)
			N.in[i.source] = i.val
			// FIXME !!!!
			N.calc()
		}
		//FIXME !!! Проблема !!! Мы из этого цикла никогда не выходим. Так что с одной сторон внешний цикл не нужен,
		//                       а с другой стороны, calc() таки будет запускаться после каждого пакета. А значит, лавинный рост не исправляется.
		fmt.Println(N, "runs calc() from listen()")
		N.calc()
		// !!! Надо сделать неблокирующий селект. Пока он может читать, пусть читает не запуская calc, а только взводя флаг, что надо сделать calc
		// Когда данные закончатся, наступает default в котором проверяется наличие этого флага. Если он стоит, то запускается calc и флаг сбрасывается.
		// Таким образом, вся обработка одного нейрона будет работать в одном потоке.
		// Надо только придумать, что делать, если данных нет. 1) Либо sleep (чтобы не было бесконечного цикла, жрущего проц) 2) либо включать блокирующий select.
	}
}

func nn_random_constructorX(n_in, n_int, n_out, max_syn int) []NeuronX {
	var n_neur int = n_in + n_int + n_out
	r := rand.New(rand.NewSource(111237))
	N := make([]NeuronX, n_in+n_int+n_out)
	for i, _ := range N {
		n := &N[i]
		n.in_ch = make(chan signal, max_syn)        // Один входной канал для всех синапсов емкостью max_syn.
		n.in = make(map[*NeuronX]float64, 1)        // Кэш входных сигналов по указателю отправителя.
		n.weight = make(map[*NeuronX]float64, 1)    // Карта весов по указателю отправителя.
		n.outs = make(map[*NeuronX]chan signal, 10) // Выходные сигналы
	}
	for i, _ := range N {
		n := &N[i]
		if i >= n_in {
			for j := 0; j <= r.Intn(max_syn); j++ { // Создаем до max_syn рэндомных синапсов.
				//n.link_withX(&N[r.Intn(n_neur)], float64(r.Intn(50))/float64(r.Intn(50)+1.0))
				n.link_withX(&N[r.Intn(n_neur)], -3.0+r.Float64()*6.0)
			}
		}
	}
	return N
}

func main() {
	var n_in = 3
	var n_int = 3
	var n_out = 3
	var max_syn = 2
	var N []NeuronX = nn_random_constructorX(n_in, n_int, n_out, max_syn)
	In := N[:n_in]
	Int := N[n_in : n_in+n_int]
	Out := N[n_in+n_int:]
	Linked := N[n_in:]

	// Работа.
	for i, _ := range Linked { // !!! Если делать 'for i,n := range Linked', то в n будет копия.
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
