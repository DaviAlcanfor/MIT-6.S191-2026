
# MIT 6.S191: Introduction to Deep Learning

* **Instrutor:** Alexander Amini
* **Aula:** Lecture 1 — Foundations of Deep Learning
* **Vídeo:** [MIT Introduction to Deep Learning | 6.S191](http://www.youtube.com/watch?v=II4giR4vOOo)

---

## 1. Inteligência Artificial, Machine Learning e Deep Learning
* **Inteligência Artificial (IA):** Construção de algoritmos artificiais capazes de processar informações para orientar decisões futuras.
* **Machine Learning (ML):** Subcampo da IA em que o computador aprende a processar informações diretamente a partir de dados, sem ser explicitamente programado passo a passo.
* **Deep Learning (DL):** Subcampo do ML focado no uso de redes neurais profundas para extrair representações hierárquicas diretamente de dados brutos (ex.: linhas e bordas $\rightarrow$ cantos e curvas $\rightarrow$ olhos/nariz $\rightarrow$ rosto completo).
* **Por que agora?** Três pilares impulsionaram o ressurgimento recente do Deep Learning:
  1. **Big Data:** Disponibilidade massiva de dados.
  2. **Hardware:** GPUs (NVIDIA, AMD) projetadas para computação paralela.
  3. **Software:** Frameworks democratizados (TensorFlow e PyTorch).

---

## 2. O Perceptron (Neurônio Único) e Forward Propagation
O bloco fundamental de qualquer rede neural é o **perceptron** (neurônio único). Ele recebe $m$ entradas ($x_1, \dots, x_m$), multiplica cada uma por seu respectivo peso ($w_1, \dots, w_m$), soma tudo com um termo de viés (*bias*, $w_0$) e aplica uma função de ativação não linear $g$:

$$\hat{y} = g\left(w_0 + \sum_{i=1}^{m} x_i w_i\right)$$

Em notação vetorial (álgebra linear):

$$\hat{y} = g\left(w_0 + \mathbf{x}^T \mathbf{w}\right)$$

Onde a soma linear antes da ativação é frequentemente denotada por $z$:

$$z = w_0 + \mathbf{x}^T \mathbf{w} \implies \hat{y} = g(z)$$

---

## 3. Funções de Ativação Não Lineares
Sem funções de ativação não lineares, qualquer rede neural — independente da profundidade — colapsaria em uma única transformação linear. Como dados do mundo real são altamente não lineares, a ativação permite que o modelo aprenda fronteiras de decisão curvas.

* **Sigmoid:** Comprime qualquer número real para o intervalo $(0, 1)$, ideal para representar probabilidades:
  $$g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
* **ReLU (Rectified Linear Unit):** Introduz não linearidade mantendo valores não negativos no intervalo $[0, +\infty)$:
  $$g(z) = \max(0, z)$$

### Exemplo Geométrico de Fronteira de Decisão
Dado um neurônio treinado com $w_0 = 1$, $\mathbf{w} = \begin{bmatrix} 3 \\ -2 \end{bmatrix}$ e ativação Sigmoid:

$$z = 1 + 3x_1 - 2x_2$$

Para a entrada $\mathbf{x} = \begin{bmatrix} -1 \\ 2 \end{bmatrix}$:

$$z = 1 + 3(-1) - 2(2) = 1 - 3 - 4 = -6$$

Como $z < 0$, o ponto está à esquerda da fronteira de decisão linear ($1 + 3x_1 - 2x_2 = 0$), resultando em $\sigma(-6) \approx 0.002 < 0.5$.

---

## 4. Arquitetura de Redes Neurais (Camadas e Profundidade)
* **Camada Densa / Linear (*Multi-Output*):** Múltiplos neurônios recebem o mesmo vetor de entrada $\mathbf{x}$, mas cada neurônio $i$ possui seu próprio vetor de pesos $\mathbf{w}^{(i)}$ e viés $w_0^{(i)}$:
  $$z_i = w_{0,i} + \sum_{j=1}^{m} x_j w_{j,i}$$
* **Single Layer Neural Network:** Possui uma camada oculta (*hidden layer*) intermediária $\mathbf{h}$ entre a entrada $\mathbf{x}$ e a saída $\hat{\mathbf{y}}$, utilizando duas matrizes de pesos ($W^{(1)}$ e $W^{(2)}$):
  $$z_i = w_{0,i}^{(1)} + \sum_{j=1}^{m} x_j w_{j,i}^{(1)}$$
  $$\hat{y}_i = g\left(w_{0,i}^{(2)} + \sum_{j=1}^{d_1} g(z_j) w_{j,i}^{(2)}\right)$$
* **Deep Neural Network (DNN):** Empilhamento sequencial de múltiplas camadas ocultas (geralmente $> 3$ camadas) com não linearidades intercaladas.

---

## 5. Funções de Perda (*Loss Functions*)
Quantificam o erro entre a previsão do modelo $f(\mathbf{x}^{(i)}; \mathbf{W})$ e o valor real (*ground truth*) $y^{(i)}$.

* **Empirical Loss (Perda Empírica Total):** Média das perdas sobre todo o conjunto de $n$ exemplos:
  $$J(\mathbf{W}) = \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}\left(f(\mathbf{x}^{(i)}; \mathbf{W}), y^{(i)}\right)$$
* **Binary Cross-Entropy Loss:** Usada em classificação binária (saídas probabilísticas entre $0$ e $1$):
  $$J(\mathbf{W}) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log\left(f(\mathbf{x}^{(i)}; \mathbf{W})\right) + (1 - y^{(i)}) \log\left(1 - f(\mathbf{x}^{(i)}; \mathbf{W})\right) \right]$$
* **Mean Squared Error (MSE):** Usada em regressão (valores contínuos):
  $$J(\mathbf{W}) = \frac{1}{n} \sum_{i=1}^{n} \left(y^{(i)} - f(\mathbf{x}^{(i)}; \mathbf{W})\right)^2$$

---

## 6. Treinamento: Gradient Descent e Backpropagation
O objetivo do treinamento é encontrar os pesos ótimos $\mathbf{W}^*$ que minimizam a perda empírica:

$$\mathbf{W}^* = \arg\min_{\mathbf{W}} J(\mathbf{W})$$

### Algoritmo Gradient Descent
1. Inicializar os pesos $\mathbf{W}$ aleatoriamente ($\sim \mathcal{N}(0, \sigma^2)$).
2. Repetir até a convergência:
   * Calcular o gradiente $\frac{\partial J(\mathbf{W})}{\partial \mathbf{W}}$.
   * Atualizar os pesos na direção oposta ao gradiente, escalados pela taxa de aprendizado (*learning rate*) $\eta$:
     $$\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{\partial J(\mathbf{W})}{\partial \mathbf{W}}$$

### Backpropagation (Regra da Cadeia)
Calcula a sensibilidade da perda $J(\mathbf{W})$ em relação a cada peso propagando derivadas da saída até a entrada usando a regra da cadeia:

$$\frac{\partial J(\mathbf{W})}{\partial w_2} = \frac{\partial J(\mathbf{W})}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_2}$$

Para camadas mais profundas (ex.: $w_1$):

$$\frac{\partial J(\mathbf{W})}{\partial w_1} = \frac{\partial J(\mathbf{W})}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$$

---

## 7. Otimização na Prática e Regularização

### Taxa de Aprendizado ($\eta$) e Otimizadores Adaptativos
* $\eta$ muito pequeno: convergência lenta e risco de ficar preso em mínimos locais.
* $\eta$ muito grande: instabilidade e divergência (*overshooting*).
* **Otimizadores adaptativos** ajustam $\eta$ dinamicamente (ex.: **Adam**, **Adadelta**, **Adagrad**, **RMSProp**).

### Mini-Batch Stochastic Gradient Descent (SGD)
* **Batch GD:** Calcula o gradiente sobre todos os $n$ dados (preciso, mas computacionalmente caro).
* **SGD puro:** Calcula o gradiente sobre $1$ ponto aleatório (rápido, mas muito ruidoso).
* **Mini-Batch GD:** Calcula o gradiente sobre um subconjunto de tamanho $B$ (ex.: $B = 32$):
  $$\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{1}{B} \sum_{k=1}^{B} \frac{\partial J_k(\mathbf{W})}{\partial \mathbf{W}}$$

### Overfitting vs. Underfitting e Regularização
* **Underfitting:** Modelo simples demais para capturar a estrutura dos dados.
* **Overfitting:** Modelo memoriza ruídos do treino e falha em generalizar para dados inéditos.
* **Regularização:** Técnicas para melhorar a generalização:
  1. **Dropout:** Durante o treino, zera aleatoriamente uma fração $p$ (ex.: $p = 0.5$) das ativações em cada *mini-batch*, forçando a rede a não depender de caminhos únicos.
  2. **Early Stopping:** Monitora a perda de treino e de teste/validação simultaneamente e interrompe o treinamento (ou recupera o *checkpoint*) no ponto em que a perda de teste começa a subir.
