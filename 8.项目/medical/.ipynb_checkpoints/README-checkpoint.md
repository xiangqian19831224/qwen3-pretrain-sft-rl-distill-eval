# Medical Doctor Agent

https://github.com/user-attachments/assets/89bb706b-3430-44d6-8912-6378edeb94d9

> This repository contains my team's final project, with a grade of **97/100**, for the **Reinforcement Learning** subject at **University of Technology Sydney** (UTS) taught by [Assoc. Prof. Nabin Sharma](https://profiles.uts.edu.au/Nabin.Sharma).

## I. Introduction

Clinical question-answering requires verifiable reasoning and machine-readable outputs, but general-purpose LLMs often produce unstructured rationales or fragile answers. We introduce a _two-stage post-training pipeline_ that transforms small LMs into structured medical reasoners:

-   First, **Supervised Fine-Tuning (SFT)** trains the response grammar, reasoning within `<THINK>…</THINK>` followed by a final medical decision in `<ANSWER>…</ANSWER>`.
-   Next, we implement **Group Relative Policy Optimization** ([GRPO](https://arxiv.org/pdf/2402.03300)) with a [multi-reward setup](#III-multi-reward-system) that simultaneously optimizes: **(i)** strict format adherence, **(ii)** partial credit for format, and **(iii)** semantic answer correctness through an [LLM verifier](https://huggingface.co/FreedomIntelligence/medical_o1_verifier_3B) that manages clinical aliases and wording differences.

We utilize LoRA for efficient parameter updates and a length-independent **Dr. GRPO** objective to prevent reward-length coupling. Evaluated on [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) (n=1,273) and [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) (n=4,183), our top model (**Qwen3-1.7B-Instruct** + [GRPO](https://arxiv.org/pdf/2402.03300)) attains 49.41% and 46.07% exact-match accuracy, respectively, with nearly 100% format compliance; [GRPO](https://arxiv.org/pdf/2402.03300) also surpasses [PPO](https://arxiv.org/abs/1707.06347) on both datasets. These findings demonstrate that verifier-guided, multi-signal [GRPO](https://arxiv.org/pdf/2402.03300) consistently enhances factual accuracy while ensuring outputs are interpretable and conform to templates, offering a practical route toward reliable, compact medical reasoning systems.

临床问答任务要求可验证的推理过程和机器可读的输出格式，但通用大语言模型往往只能生成非结构化的推理文本或稳定性不足的答案。为此，我们提出了一种两阶段的后训练（post-training）流程，用于将小规模语言模型转化为结构化的医学推理模型。

首先，在**监督微调（Supervised Fine-Tuning，SFT）**阶段，我们训练模型的响应语法，使其推理过程严格封装在 <THINK>…</THINK> 标签中，并在 <ANSWER>…</ANSWER> 标签中给出最终的医学决策结果。

其次，我们引入了群体相对策略优化（Group Relative Policy Optimization，GRPO），并设计了一个多重奖励机制，同时优化以下目标：
(i) 严格的格式遵循度；
(ii) 格式部分正确的软奖励；
(iii) 语义层面的答案正确性，该项通过一个 LLM 评估器（verifier）实现，用于处理医学同义表达与措辞差异。

在训练过程中，我们采用 LoRA 进行高效的参数更新，并引入与生成长度无关的 Dr. GRPO 目标函数，以避免奖励与输出长度之间的耦合问题。

在 MedQA-USMLE（n=1,273） 和 MedMCQA（n=4,183） 数据集上的评测结果表明，我们的最佳模型 Qwen3-1.7B-Instruct + GRPO 分别取得了 49.41% 和 46.07% 的精确匹配（Exact Match）准确率，同时格式合规率接近 100%；此外，GRPO 在两个数据集上均显著优于 PPO。

## II. Proposed Solution

![](./images/solution.png)

The models will be fine-tuned to produce structured outputs with a reasoning section wrapped in `<THINK>` tags for step-by-step logic, followed by a precise medical answer in `<SOLUTION>` tags. We designed a two-stage pipeline here, **Supervised Fine-Tuning (SFT)** followed by **Reinforcement Learning (RL)**, to transform LLMs into structured medical reasoners:

-   **Phase 1 - SFT**: The goal here is not to teach the model to be a medical solver yet. It's to teach the model the grammar of our desired output. Here, we used a [dataset](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) of multiple medical problems with high-quality reasoning traces, formatted with our custom `<THINK>` and `<SOLUTION>` tags. This forces the model to learn the structural template we defined.
-   **Phase 2 - RL**: This is where we refine the logic using **RL**. Now that the model already knows how to structure its response, we then use [GRPO](https://arxiv.org/pdf/2402.03300) and this [medical questions dataset](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-verifiable-problem) to teach it how to reason accurately and arrive at the correct medical answer.

[GRPO](https://arxiv.org/pdf/2402.03300) is an SOTA RL technique designed to overcome key limitations of the traditional [PPO](https://arxiv.org/abs/1707.06347). Specifically, [PPO](https://arxiv.org/abs/1707.06347) can suffer from high memory overhead due to its reliance on value network and instability in value function estimation. [GRPO](https://arxiv.org/pdf/2402.03300) addresses these issues by eliminating the need for a learned value function, instead using **group-relative advantage estimation** across multiple responses. This not only reduces computational cost but also improves training stability and scalability

Another drawback of [PPO](https://arxiv.org/abs/1707.06347) is that it also relies on a reward model that assigns **an absolute score to a generation**. There are 2 problems with this:

-   First, the reward model can rely on human judgments that usually lack explicit criteria and require expensive human annotation.
-   Second, it can be unstable as the LLM might learn to **hack** the reward. For example, it can generate very long completion if the length is correlated with a higher score. The solution here is to define a list of smaller verifiable rewards, not a final all consuming singular one.

With [GRPO](https://arxiv.org/pdf/2402.03300), we already generated **a group of responses** for each prompt right? Instead of scoring each one in isolation, we evaluate them relative to each other with our [multi-reward system](#III-multi-reward-system):

-   This **Reinforcement Learning with Verifiable Rewards** will allow us to further eliminate the need for a reward model and replace subjective human evaluation with reliable, objective signals.
-   This relative comparison is far more stable and directly optimizes for what we want: better reasoning, not just a higher score.

这些模型将通过微调来生成结构化输出：先在 <THINK> 标签中给出逐步推理过程，再在 <SOLUTION> 标签中给出精确的医学答案。
我们设计了一个两阶段训练流水线——先进行监督微调（SFT），再进行强化学习（RL），以将通用大语言模型转化为具备结构化医学推理能力的模型。

阶段一：监督微调（SFT）

这一阶段的目标并不是教模型如何真正解决医学问题，而是教会模型我们期望的输出语法和结构。
在这一阶段，我们使用了一个包含多种医学问题的数据集，其中配有高质量的推理轨迹，并统一采用自定义的 <THINK> 和 <SOLUTION> 标签进行格式化。
这样可以强制模型学习并内化我们所定义的输出模板和结构规范。

阶段二：强化学习（RL）

在模型已经掌握输出结构之后，第二阶段通过强化学习来提升其推理质量和答案正确性。
我们使用 GRPO（Group Relative Policy Optimization） 以及医学问答数据集，对模型进行进一步训练，使其能够更准确地进行推理并得到正确的医学结论。

GRPO 是一种先进的强化学习方法，旨在克服传统 PPO 的关键局限性。
具体来说，PPO 由于依赖价值网络（value network），通常存在以下问题：
    1.显存和计算开销较大
    2.价值函数估计不稳定，容易影响训练稳定性
        GRPO 通过完全移除对价值函数的依赖，改为在同一问题下生成的多条回答之间进行相对优势估计，有效解决了上述问题。这不仅降低了计算成本，还显著提升了训练的稳定性和可扩展性。
    3.PPO 的另一项缺陷与 GRPO 的改进
        PPO 还依赖于一个奖励模型（Reward Model），为每条生成结果打一个绝对分数，而这本身存在两个严重问题：
        奖励模型往往依赖人工标注，而人类评估通常缺乏明确、可执行的评价标准，且标注成本极高。
        奖励不稳定，容易被模型“投机取巧”。例如，如果奖励与生成长度相关，模型可能会倾向于输出冗长但低质量的内容。
        解决思路是：
            不使用单一、笼统的奖励信号，而是设计多个可验证的、细粒度的奖励函数。
            基于 GRPO 的相对奖励与可验证奖励
            在 GRPO 中，我们会为同一个问题生成一组候选回答。
            与其对每条回答进行孤立评分，不如在同组回答之间进行相对比较，并结合我们设计的多重可验证奖励机制进行评估。
            这种基于可验证奖励的强化学习（Reinforcement Learning with Verifiable Rewards），可以进一步消除对奖励模型的依赖，用客观、稳定、可自动验证的信号取代主观的人类评估。
            这种相对比较方式更加稳定，也更直接地优化了我们真正关心的目标：
                👉 更好的推理能力，而不仅仅是更高的奖励分数。


## III. Multi-Reward System


![](./images/rewards.png)

Our core innovation is this multi-reward design. A single reward is not enough to capture the nuances of good medical reasoning. We designed a **panel of 4 expert judges** working in parallel, each evaluating the model's output from a different perspective:

1.  The first is the **Strict Formatter** which strictly evaluate format compliance to enforce the structure. It gives a large reward only if the entire response perfectly adheres to our `THINK` and `ANSWER` structure.

2.  The second is the **Partial Formatter** giving partial credit for incomplete tags. If the model messes up the full structure, but for example, still includes the `</THINK>` tag correctly, it still gets a small amount of credit.

3.  The third, also the **most important one**. It will check if the answer in the `<ANSWER>` tag is correct or not. Given the prevalence of aliases in the medical domain, exact matching methods, which commonly applied in mathematics, will be impractical here. Instead, as suggested by [HuatuoGPT-o1](https://arxiv.org/pdf/2412.18925), we use an [LLM verifier](https://huggingface.co/FreedomIntelligence/medical_o1_verifier_3B) here and prompt it to perform validation, returning a probability of how close the prediction aligns with the ground-truth answer. We designed this function to be sophisticated, giving full marks for an high probability, partial credit for close approximations, and a heavy penalty for wrong answers to avoid overconfidence.

By combining these 3 signals, we can prevent over-optimization on 1 aspect, which can lead to reward hacking problem. The [GRPO](https://arxiv.org/pdf/2402.03300)'s group-relative policy can navigate the complex trade-offs between formatting, correctness, readability, and optimizes by ranking completions, leading to a much more capable and reliable reasoning model.

我们的核心创新在于多奖励（multi-reward）设计。单一奖励信号不足以刻画高质量医学推理中蕴含的复杂细节。因此，我们设计了一组并行工作的四位“专家评审”，从不同维度对模型输出进行评估：
第一位评审是严格格式评审（Strict Formatter），用于严格检查输出是否完全符合我们预定义的结构规范。只有当模型的完整回答严格遵循 <THINK> 与 <ANSWER> 的结构要求时，才会给予较高奖励。
第二位评审是部分格式评审（Partial Formatter），用于对不完整但部分合规的结构给予“部分奖励”。例如，当模型未能完全遵守结构规范，但仍正确包含了 </THINK> 等关键标签时，仍可获得一定的正向反馈。
第三位评审也是最重要的一位，负责评估答案的正确性。
在医学领域中，由于同义词、别名和表达多样性广泛存在，数学领域常用的精确匹配方法并不适用。受 HuatuoGPT-o1 的启发，我们采用了一个 LLM 验证器（LLM-based verifier） 来完成这一任务。该验证器会评估模型在 <ANSWER> 标签中的预测结果与标准答案之间的一致程度，并输出一个概率值。
该奖励函数被精心设计：当一致性概率较高时给予满分奖励，对接近正确的预测给予部分奖励，而对于明显错误的答案施加显著惩罚，以避免模型产生过度自信的错误判断。
通过融合这三类奖励信号，我们能够有效避免模型在单一维度上过度优化，从而减少奖励投机（reward hacking）问题的发生。
借助 GRPO 的组相对策略优化机制（group-relative policy），模型可以在格式规范性、答案正确性、可读性等多种目标之间进行权衡，并通过对多个候选生成结果进行排序来完成优化，最终得到一个更强大、更稳定且更可信的医学推理模型。

## IV. GRPO Objective Improvement

![](./images/grpo.png)

Compared to the original formulation in the [DeepSeekMath](https://arxiv.org/pdf/2402.03300) paper, we followed [Hugging Face's GRPO guideline](https://huggingface.co/docs/trl/main/en/grpo_trainer#computing-the-loss) and made some further improvements to the [GRPO](https://arxiv.org/pdf/2402.03300) objective for more efficient training:

-   First, we can calculate the _mean_ at the _group_ and the _std_ at the _batch_ level. This scaling strategy enables more robust reward shaping, as evident by this [paper](https://huggingface.co/papers/2508.08221).
-   Second, we didn't use the **KL divergence** term, as motivated by several recent studies, which showed that **KL** term is not essential for training with [GRPO](https://arxiv.org/pdf/2402.03300). Therefore, it has become a common practice to exclude it.
-   Lastly, this [paper](https://huggingface.co/papers/2503.20783) has demonstrated that the initial [GRPO](https://arxiv.org/pdf/2402.03300) formulation introduces a **response length bias**. To solve that, they proposed dividing by a **constant generation budget** instead of the sequence length, so we employ this [Dr.GRPO](https://huggingface.co/papers/2503.20783) loss here to further enhances stability by preventing the model from being biased towards longer or shorter answers, focusing purely on the quality of the content.

相比 DeepSeekMath 论文中的原始公式，我们遵循了 Hugging Face 的 GRPO 指南，并对 GRPO 目标函数做了一些进一步的改进，以提升训练效率：
首先，我们在 group 级别 计算均值，在 batch 级别 计算标准差。这种缩放策略能够带来更稳健的奖励塑形效果，相关结论也已在该论文中得到验证。
其次，我们没有使用 KL 散度项。这一做法受到多项近期研究的启发，这些研究表明，在 GRPO 训练中 KL 项并非必需，因此在实践中将其移除已成为一种常见做法。
最后，有研究指出，最初的 GRPO 公式会引入对生成长度的偏置。为了解决这一问题，论文提出用一个固定的生成预算而不是序列长度进行归一化。因此，我们在这里采用了 Dr.GRPO loss，通过避免模型偏向生成更长或更短的回答，进一步提升了训练稳定性，使模型专注于内容质量本身。

## V. Experimental Results

![](./images/results.png)

👉 You can refer to our [slides](./slides.pdf) and [full report](./report.pdf) for more details on the methodology and results analysis.

## VI. References

**1. [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1)**:

-   We adapted many ideas from their work on medical reasoning with LLMs.
-   We used their [PPO](https://arxiv.org/abs/1707.06347) approach as a baseline to compare against our [GRPO](https://arxiv.org/pdf/2402.03300) solution. Note that, we used smaller model here due to computational constraints on Colab Pro Environment.

**2. Hugging Face's Cookbook**:

-   [GRPO Trainer Documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer#computing-the-loss).
-   [Post training an LLM for reasoning with GRPO in TRL](https://huggingface.co/learn/cookbook/fine_tuning_llm_grpo_trl).
-   [HuatuoGPT-o1 Medical RAG and Reasoning](https://huggingface.co/learn/cookbook/medical_rag_and_reasoning): We followed this to build our demo with RAG capabilities.

**3. Unsloth Documentation**:

> We mainly used [Unsloth](https://docs.unsloth.ai/) to implement our [GRPO](https://arxiv.org/pdf/2402.03300) training.

-   [Reinforcement Learning (RL) Guide](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide).

-   [GRPO (Reasoning RL) notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks): We learned a lot from these notebooks.
