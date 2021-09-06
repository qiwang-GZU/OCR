<h1 id='PANet'>PANet</h1>

>## 目录
+ [PANet简介](#Abstract)
+ [相关介绍](#Introduction)
+ [相关工作](#RelatedWork)
    + [1.总体架构](#OverallArchitecture)
    + [2.功能金字塔增强模块](#FeaturePyramidEnhancementModule)
    + [3.功能融合模块](#FeatureFusionModule)
    + [4.像素聚合](#PixelAggregation)
    + [5.损失函数](#LossFunction)


><h2 id='Abstract'> PSENet简介 </h2>
[PANet](https://arxiv.org/pdf/1908.05900.pdf)全称为Pixel Aggregation Network，即像素聚合网络。它配备了低计算成本的分割头和可学习的后处理。分割头由特征金字塔模块(Feature Pyramid Enhancement Module, FPEM)和特征融合模块( Feature Fusion Module, FFM)组成。FPEM是一个可级联的U型模块，它可以引入多层次的信息来指导更好的分割。FFM可以将FPEM给出的不同深度的特征收集成最终的特征进行分割。可学习的后处理是由像素聚合( Pixel Aggregation，PA)实现的，它可以通过预测的相似性向量精确地聚合文本像素。

><h2 id='Introduction'> 相关介绍 </h2>
随着近年来基于CNN的目标检测和分割的发展，场景文本(scene text detection)检测取得了很大的进展。任意形状文本检测(Arbitrary-shaped text detection)是文本检测中最具挑战性的任务之一，也提出了检测曲线文本实例的新方法。但是大部分方法仍然存在缺陷：  
+ 推理速度低或复杂的后处理步骤从而限制了在现实环境中的部署
+ 具有高效率的文本检测器大多是为四边形文本实例设计的，它们在检测弯曲文本时存在缺陷

PANet是可以在速度与表现之间取得一个较好的平衡的任意形状的文本检测器，它拥有非常简单的流水线，仅仅只需两步：
+ 利用分割网络预测文本区域、核和相似度向量。
+ 从预测的内核中重建完整的文本实例

<div align=center><img src="imgs/PANet/1.png" alt='实例'></div>

为了高效，我们需要减少这两个步骤的时间成本。首先，分割需要一个轻量级的主干,本文采用ResNet18作为PAN的默认主干网络。然而，轻量级主干在特征提取方面相对较弱，因此其特征通常具有较小的接受域和较弱的表示能力。为了弥补这一缺陷，我们提出了一个由两个模块组成的低计算成本的分割头：
+ FPEM：FPEM是一个由可分卷积构建的U形模块。因此，因此FPEM能够通过融合低级和高级信息和最小的计算开销来增强不同尺度的特征。此外，FPEM是可级联的，这允许我们通过在后面添加FPEM来补偿轻量级主干的深度。
+ FFM：为了收集低级和高级的语义信息，在最终分割之前引入了FFM来融合不同深度的FPEMs生成的特征
+ PA：为了准确地重建完整的文本实例，本文提出了一种可学习的后处理方法，即像素聚合(Pixel Aggregation， PA)，它可以引导文本像素通过预测的相似性向量来纠正内核

><h2 id='RelatedWork'> 相关工作 </h2>

近年来，基于深度学习的文本检测器取得了显著的效果。这些方法大多大致可分为两类：基于锚的方法和无锚的方法。
|  基于锚定的文本探测器   | 无锚文本探测器  | 实时文本检测
|  ----  | ----  | --- |
| Faster R-CNN  | PSENet | EAST|
|  SSD  | PixelLink | MCN|
| TextBoxes | EAST|
|TextBoxes++|DeepReg|
| RRD |TextSnake|
| SSTD | 
| RRPN | 
| SPCNet |
|Mask Text Spotter|
|Mask R-CNN|

><h2 id='RelatedWork'> 相关工作 </h2>
<h3 id='OverallArchitecture'>1.总体架构</h3>
<div align=center><img src="imgs/PANet/2.png" alt='实例'></div>

本文采用轻量级模型(ResNet-18)作为PAN的主干网络。主干网络的conv2、conv3、conv4和conv5阶段生成四个特征图，并且他们相对输入图像的步长为4、8、16、32像素。我们使用1×1卷积，将每个特征映射的通道数减少到128，并得到一个薄的特征金字塔$Fr$。特征金字塔被$n_c$级联FPEM增强。每个FPEM产生一个增强的特征金字塔，因此有$n_c$增强的特征金字塔$F^1，F^2，...，F^{n_c}$。FFM将$n_c$增强的特征金字塔融合到一个特征地图$F_f$中，其步幅为4像素，通道数为512。$F_f$用于预测文本区域、核和相似度向量。最后，我们应用一种简单而高效的后处理算法来获得最终的文本实例。

<h3 id='FeaturePyramidEnhancementModule'>2.功能金字塔增强模块</h3>
<div align=center><img src="imgs/PANet/3.png" alt='实例'></div>

FPEM是一个U型模块，它由向上增强和向下增强两个阶段组成。向上增强作用于输入特征金字塔。在此阶段中，在具有32、16、8、4像素步长的特征映射上迭代执行增强。在下行阶段，输入是由上行增强产生的特征金字塔，增强从4步到32步进行增强。  
同时，向下尺度增强的输出特征金字塔是FPEM的最终输出。我们使用可分离卷积(separable convolution)（3×3 深度卷积(depthwise convolution)，然后是1×1投影(projection)），而不是用常规的卷积来构建FPEM的连接部分$\bigoplus$(上图虚线框）。因此，FPEM能够以较小的计算开销扩大接受域（3×3深度卷积)和深化网络(1×1卷积）。  
与FPN[(Feature Pyramid Networks for Object Detection)](https://arxiv.org/pdf/1612.03144.pdf)类似，FPEM能够通过融合低级和高级信息来增强不同尺度的特性。此外，与FPN不同的是，FPEM还有另外两个优点:
+ FPEM是一个可级联的模块。随着级联数$n_c$的增加，不同尺度的特征图被更充分地融合，特征的接受域变大。
+ FPEM的计算成本很便宜。FPEM是由只需最小的计算的可分离卷积构成，FPEM所需浮点运算仅为FPN的五分之一。

<h3 id='FeatureFusionModule'>3.功能融合模块</h3>
<div align=center><img src="imgs/PANet/4.png" alt='实例'></div>

采用特征融合模块来融合不同深度的特征金字塔$F^1,F^2,\cdots,F^{n_c}$。因为低级和高级的语义信息对于语义分割都很重要。结合这些特征金字塔的一个直接和有效的方法是对它们进行上采样和连接。然而，该方法给出的融合特征图具有较大的信道数$(4\times 128\times n_c)$，从而降低了最终预测的速度。因此，本文提出了另一种融合方法，首先通过元素加法来组合相应尺度的特征图。然后对添加后的特征图进行上采样，并连接到只有4个×128个通道的最终特征图中。

<h3 id='PixelAggregation'>4.像素聚合</h3>

文本区域保持了文本实例的完整形状，但紧密靠近的文本实例的文本区域经常重叠。相反，使用内核可以很好地区分文本实例。但内核并不是完整的文本实例，要重建完整的文本实例，我们需要将文本区域中的像素合并到内核中。本文提出了一种可学习的算法，即像素聚合(Pixel Aggregation)，以引导文本像素走向正确的核。
在像素聚合中，我们借用聚类的想法来从内核中重建完整的文本实例。让我们将文本实例视为集群。文本实例的核心是集群中心。文本像素是要聚类的样本。当然，要将文本像素聚合到相应的内核中，同一文本实例的文本像素和内核之间的距离应该很小。在训练阶段，我们使用聚合损失$L_{agg}$来实现：
$$L_{agg}={1\over N}{ \sum \limits_{i=1}^N}{1\over |T_i|}{\sum \limits_{p\in T_i}}{\ln(D(p,K_i)+1)} $$
$$D(p,K_i)={\max(||F(p)-G(K_i)||-\delta_{agg}, 0)}^2$$
其中，$N$是文本实例的数量。$T_i$是第i个文本实例。$D(p,K_i)$定义文本像素$p$与文本实例的内核$K_i$之间的距离$T_i$。$\delta_{agg}$是一个常数，通过实验设置为0.5来过滤容易的样本。$F(p)$是像素的相似度向量$p$。$G(\cdot)$是内核的相似性向量$K_i$,可以通过$\sum_{q\in K_i}F(q)/|K_i|$。
此外，集群中心还需要保持歧视(discrimination)。因此，不同文本实例的内核应该保持足够的距离。我们使用歧视损失$L_{dis}$在训练期间实现这一规则:
$$L_{dis}={1\over N(N-1)}{\sum \limits _{i=1}^N}{\sum_{j=1,x\neq y}^{N}}{ln(D(K_i,K_j)+1)}$$
$$D(K_i,K_j)=max{(\delta_{dis}-||G(K_i)-G(K_j)||,0)}^2$$
${L_{dis}}$试图保持内核之间的距离不小于${δ_{dis}}$，在我们所有的实验中都设置为3。在测试阶段，我们使用预测的相似度向量来引导文本区域中的像素到相应的核。详细的后处理步骤如下：
+ 在内核的分割结果中找到连接的组件，每个连接的组件都是一个内核
+ 对于每个核$K_i$，有条件地将其相邻的文本像素（4路）$p$合并在预测的文本区域中，而其相似性向量的欧氏距离小于$d$
+ 重复步骤二，直到没有合格的邻居文本像素。

<h3 id='LossFunction'>5.损失函数</h3>

损失函数可表示为：
$$L=L_{tex}+\alpha L_{ker}+\beta(L_{agg}+L_{dis})$$
其中$L_{tex}$是文本区域的丢失，$L_{ker}$是内核的丢失。α和β用于平衡$L_{tex}$、$L_{ker}$、$L_{agg}$和$L_{dis}$之间的重要性，我们在所有实验中将它们分别设置为0.5和0.25。
考虑到文本和非文本像素的极端不平衡，我们遵循采用骰子损失(dice loss)来监督文本区域的分割结果$P_{tex}$和内核的$P_{ker}$。因此，$L_{tex}$和$L_{ker}$可以写如下：
$$L_{tex}=1-{{2 {\sum \nolimits _i}P_{tex}(i)G_{tex}(i)}\over {\sum \nolimits _i}P_{tex}(i)^2+{\sum \nolimits _i}G_{tex}(i)^2}$$
$$L_{ker}=1-{{2 {\sum \nolimits _i}P_{ker}(i)G_{ker}(i)}\over {\sum \nolimits _i}P_{ker}(i)^2+{\sum \nolimits _i}G_{ker}(i)^2}$$
其中，$P_{tex}(i)$和$G_{tex}(i)$分别指分割结果中第$i$个像素的值和标准文本区域。标准文本区域为二进制图像，其中文本像素为1，非文本像素为0。同样，$P_{ker}(i)$和$G_{ker}(i)$是指预测中的第i个像素值和内核的标准值。核的标准值是通过收缩原始标准多边形生成的，我们按照[Shape Robust Text Detection with Progressive Scale Expansion Network](https://arxiv.org/pdf/1903.12473.pdf)中的方法将原始多边形收缩的比例为$r$.注意，在计算$L_{tex}$时，我们使用在线硬例挖掘[(Online Hard Example Mining, OHEM)](https://arxiv.org/pdf/1604.03540.pdf)忽略简单的非文本像素，在计算$L_{ker}$、$L_{agg}$和$L_{dis}$时只考虑地面真实中的文本像素。