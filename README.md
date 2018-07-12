# PersonBodyType
Classify people by body type.

## Test One
1. DataSet Description<br>
1\) Divid by BMI.<br>
&emsp;Divided into three categories by body type based on bmi. Methods as below.
> 编号&emsp;分类&emsp;BMI 范围<br>
> &emsp;0&emsp; 偏瘦&emsp; <= 18.4<br>
> &emsp;1&emsp; 正常&emsp; 18.5 ~ 23.9<br>
> &emsp;2&emsp; 过重&emsp; 24.0 ~ 27.9<br>
> &emsp;3&emsp; 肥胖&emsp; >= 28.0<br>

2. NetWork<br>
&emsp;Resnet+Dense+Softmax in Keras.

3. Problem<br>
1\) Can't Learning the GT.<br>
&emsp;The Train loss lower than Val loss, and the Tarin acc higher than Val acc. The Train acc may be 90% but the Val acc is only 30% or less. The Train acc is error but I don't how it calculate. The acc is low while I calculate the acc by loading the model after training.<br>
&emsp;May be due to unreasonable data partitioning. Because the fitness training person looks very small even if they have high bmi.
&emsp;Analyze the calculation method of acc or Change the way the body is divided. Using shirtsize or pantsize to divid the body type.