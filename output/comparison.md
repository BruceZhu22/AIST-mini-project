# OFT Finetuning — Before vs. After Comparison

**Base model**: `Qwen/Qwen2-0.5B-Instruct`  
**OFT adapter**: `./output/qwen2-0.5b-oft`

---

## Example 1

**Instruction:** Explain what orthogonal fine-tuning (OFT) is and how it differs from LoRA.

**Base model response:**

> Orthogonal Fine-Tuning (OFT) is a technique used in deep learning to improve the performance of neural networks by adjusting the weights of the layers in such a way that they are more closely aligned with each other, leading to better generalization and reduced overfitting.   
>   
> LoRA stands for "Learning on Random Access", which refers to the process of training a model using random access rather than fixed inputs. In contrast, OFT adjusts the weights of the layers in a neural network so that they are closer to each other, resulting in improved generalization and reduced overfitting compared to LoRA.  
>   
> In summary, orthogonal fine-tuning (OFT) involves adjusting the weights of the layers in a neural network to be more closely aligned with

**OFT-finetuned response:**

> Orthogonal Fine-Tuning (OFT) is a technique used in deep learning to improve the performance of neural networks by reducing the number of parameters needed for each layer. It works by adjusting the weights of the layers so that they are optimized to minimize the loss function, while also ensuring that the network can learn the correct features without overfitting. This approach allows the model to be trained on a smaller amount of data than traditional methods, which can lead to faster training times and better results. Additionally, it can help reduce the computational cost of training the model, as it does not require the entire dataset to be processed at once. Overall, OFT is a powerful technique that can significantly improve the performance of neural networks.

---

## Example 2

**Instruction:** Write a Python function that computes the Fibonacci sequence up to n.

**Base model response:**

> ```python  
> def fibonacci(n):  
>     if n <= 0:  
>         return []  
>     elif n == 1:  
>         return [0]  
>     elif n == 2:  
>         return [0, 1]  
>       
>     fib_sequence = [0, 1]  
>     for i in range(2, n):  
>         next_number = fib_sequence[i-1] + fib_sequence[i-2]  
>         fib_sequence.append(next_number)  
>           
>     return fib_sequence  
> ```  
>   
> This Python function takes an integer `n` as input and returns a list of integers representing the Fibonacci sequence up to `n`. The function first checks if `n` is less than or equal to 0, in which case it returns an empty list. If

**OFT-finetuned response:**

> def fibonacci(n):  
>     if n <= 0:  
>         return []  
>     elif n == 1:  
>         return [0]  
>     else:  
>         fib_sequence = [0, 1]  
>         for i in range(2, n):  
>             next_number = fib_sequence[i-1] + fib_sequence[i-2]  
>             fib_sequence.append(next_number)  
>         return fib_sequence  
>   
> # Example usage  
> print(fibonacci(8)) # Output: [0, 1, 1, 2, 3, 5, 8]   
> print(fibonacci(10)) # Output: [0, 1, 1, 2, 3, 5, 8, 13,

---

## Example 3

**Instruction:** Translate the following English sentence to French.

**Input:** Artificial intelligence is transforming every aspect of modern life.

**Base model response:**

> The translation of the given English sentence "Artificial intelligence is transforming every aspect of modern life" into French would be: "L'intelligence artificielle transforme chaque aspect de la vie moderne."   
>   
> This sentence conveys the same meaning as the original English sentence but in a different language. It highlights the significant impact of AI on various aspects of modern life. The use of "transformer" and "every aspect" emphasizes the broad scope of its influence. Additionally, the phrase "modern life" suggests that this transformation is happening within the current era or period. Overall, the sentence effectively communicates the essence of the situation being described.

**OFT-finetuned response:**

> L'intelligence artificielle transforme chaque aspect de notre vie moderne.   
>   
> This translates to: "L'intelligence artificielle transforme chaque aspect de notre vie moderne." in French. It means that artificial intelligence is changing almost everything we do today. The word "transforme" means to change or make something better. The phrase "every aspect of our life" refers to all aspects of our lives, including work, leisure, and personal relationships. The last part of the sentence, "modern," suggests that it's about the future. Overall, this sentence conveys the idea that artificial intelligence is revolutionizing many aspects of our lives.

---

## Example 4

**Instruction:** Summarise the following paragraph in one sentence.

**Input:** Transformers, introduced in 'Attention Is All You Need' (Vaswani et al., 2017), replaced recurrent architectures with self-attention mechanisms, enabling highly parallelisable training and yielding state-of-the-art results across NLP tasks.

**Base model response:**

> The Transformers introduced in 'Attention Is All You Need' (Vaswani et al., 2017) significantly transformed the way NLP tasks are trained by replacing recurrent architectures with self-attention mechanisms, leading to highly parallelisable training methods that yield state-of-the-art results across various NLP tasks.

**OFT-finetuned response:**

> Self-attention mechanisms were used to replace recurrent architectures in Transformers, leading to highly parallelisable training and achieving state-of-the-art results across NLP tasks. This approach significantly improves performance on various NLP tasks.

---

## Example 5

**Instruction:** Convert 98.6 degrees Fahrenheit to Celsius.

**Base model response:**

> To convert 98.6 degrees Fahrenheit to Celsius, you need to multiply it by 1.8. Here's how:  
>   
> 98.6 * 1.8 = 175.24  
>   
> Therefore, 98.6 degrees Fahrenheit is equivalent to 175.24 degrees Celsius.

**OFT-finetuned response:**

> 98.6 degrees Fahrenheit is equivalent to 37.5 degrees Celsius.   
>   
> Note: The conversion factor between Fahrenheit and Celsius is 1 degree Celsius = 5/9 degrees Fahrenheit. Therefore, 98.6 degrees Fahrenheit is equivalent to 37.5 degrees Celsius.   
>   
> Please note that this conversion may vary slightly depending on the specific temperature measurement system used. For example, some systems use Kelvin instead of Celsius. If you need to convert from one system to another, it's always best to check the specific requirements for your application or project.

---

## Example 6

**Instruction:** Give three practical tips for improving sleep quality.

**Base model response:**

> Improving sleep quality can be achieved through several practical tips. Here are three suggestions:  
>   
> 1. **Establish a Sleep Schedule**: Try to go to bed and wake up at the same time every day, even on weekends. This helps regulate your body's internal clock and ensures you get enough restful sleep each night.  
>   
> 2. **Create a Bedtime Routine**: Establish a bedtime routine that includes relaxation techniques like reading or listening to calming music before going to bed. This helps signal to your body that it's time to wind down and prepare for sleep.  
>   
> 3. **Limit Screen Time Before睡觉**: Avoid using electronic devices right before bed as they can interfere with your natural sleep-wake cycle. Use apps or other tools designed to help you fall asleep more

**OFT-finetuned response:**

> 1. Establish a regular sleep schedule: Try to go to bed and wake up at the same time every day.  
> 2. Create a relaxing bedtime routine: Before going to bed, try to do something calming like reading or listening to music.  
> 3. Avoid caffeine and alcohol before bed: These substances can interfere with your body's natural sleep-wake cycle. If you must consume them, try to avoid in the hours leading up to bedtime.   
> 4. Keep your bedroom cool and dark: This will help regulate your body's temperature and make it easier to fall asleep. Also, use blackout curtains or blinds to block out any light sources.   
> 5. Exercise regularly: Regular physical activity can improve your mood and energy levels, which can lead to better

---

