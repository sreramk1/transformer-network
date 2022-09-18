# Here is why the transformer network's positional embeddings and encodings feel like a hack

*Sreram K (sreramk26@gmail.com)*

*Date: 18-09-2022*


## The transformer network's positional embedding and encoding are a "hack" because of the following: 


If you assign a embedding vector $a_{j}$ to the English word "Hello", then, for every positional embedding or encoding $p_i$ you will have, 

$$a'_{j,1} = a_{j} + p_{1}, a'_{j,2} = a_j + p_{2}, ..., a'_{j, n} = a_j + p_n$$ 

Do you see that we don't have one embedding vector for each word, instead we have many? 

**So if the network does not see all the possible positional embeddings/encodings of the word, then it wouldn't have leaned to properly associate the word "Hello" with the other words in the sentence.**

Instead of having one embedding vector for "Hello", we now have multiple vectors! "Hello" at a different position acts like a different embedding vector. 

LSTM and other recurrent neural networks don't have this problem. They deal with the **actual** embeddings instead of encoded embeddings. So, you precisely have one embedding vector for the word "hello" for every position!

*Having a different vector for every position of a specific word, might cause the model to memorize the word along with its position. So that's why OpenAI's GPT-3 is just a compressed information storage with an advanced form of indexing (of a kind we have never seen before).*

## Let's run through the math to see how embeddings and encodings behave:

Let me show you why transformer network's positional embedding and encoding are just a hack by running through their math. 

Let $\vec{x_i}$ be the embedding vector of the word $i$ in the input sentence. 

Then, the sequence formed by $\vec{y_i}$ for all $i$ will be the output of the transformer layer. 

I.e., $\text{transformerLayer}(\{ \vec{x_1}, ..., \vec{x_M}\}) = \{\vec{y_1}, ..., \vec{y_M}\}$

If we assume $M$ to be the length of the sentence, then $1\leq i \leq M$. 

The symbols, $W_q$, $W_k$ and $W_v$ represents the weight values for query, key and value of the transformer layer. 

$\vec{q_i} = W_q \vec{x_i}$ &emsp;&emsp;... (1)

$\vec{k_i} = W_k \vec{x_i}$ &emsp;&emsp;... (2)

$\vec{v_i} = W_v \vec{x_i}$ &emsp;&emsp;... (3)

$h_{ij} = \vec{q_i}^{T} \vec{k_j}$ &emsp;&emsp;... (4)

$\vec{y_i} = \sum_{j} \text{softmax}(h_{ij})\vec{v_j}$ &emsp;&emsp;... (5)


Now, in these above expressions, let's replace $x_i$ with $x'_i + p_i$. Here, $p_i$ is either a positional embedding or a positional encoding. My explanation will valid regardless. 

So, we have, 

$\vec{q_i} = W_q (\vec{x'_i} + \vec{p_i} ) = W_q \vec{x'_i} + W_q\vec{p_i}$ &emsp;&emsp;... (6)

$\vec{k_i} = W_k (\vec{x'_i} + \vec{p_i} ) = W_k \vec{x'_i} + W_k\vec{p_i}$ &emsp;&emsp;... (7)

$\vec{v_i} = W_v (\vec{x'_i} + \vec{p_i} ) = W_v \vec{x'_i} + W_v\vec{p_i}$ &emsp;&emsp;... (8)

$$h_{ij} = \vec{q_i}^{T} \vec{k_j} =  (W_q \vec{x'_i})^T W_k\vec{x'_j} +  (W_q \vec{x'_i})^T W_k\vec{p_j} +  (W_q\vec{p_i})^TW_k \vec{x'_j} +    (W_q\vec{p_i})^TW_k\vec{p_j} $$ 

&emsp;&emsp;... (9)


Now, (9) is finally used in (5).  

Observe (9) carefully. If you write the sum $\sum_{i}\sum_{j}h_{i,j}$, then, you will have the **exact same value even if you jumble the position of the words in the input sentence.**

So, at this point, you are forced to introduce some kind of non-linearity. $\text{softmax}$ is non-linear and it makes perfect sense to be introduced at this point.

When you *do* introduce softmax for its non-linearity and its ability to serve as the perfect coefficient for the $v_i$ values in the sum, you are transforming the base word embedding $x'_i$  into something else $x_i$. 

The same word in the sentence will have more than one embedding for each position. And this might cause the model to treat $a_{i} + p_{j}$ and $a_{i} + p_{k}$ differently. Where, $a_{i}$ is an embedding of a specific English word. And $p_{i}$ and $p_{k}$ are embedding or encoding vectors for position $j$ and $k$ respectively.

This is why it feels like a hack! The word "Hello" in English, must not not be represented differently based on its position in the sentence. 

