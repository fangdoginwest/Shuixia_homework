# Shuixia_homework
## 训练模型
*test.py*是检测PC是否可以用N卡训练
接着用*train.py*训练模型*usept.py*是检测模型的检测成功率
## 转换模型
然后用*exchange.py*把.pt模型变成.ONNX模型
接着使用*jiaozhun.py*生成校准数据用于ACT量化时校准
## 部署模型
最后在华为开发板中使用ACT量化转化成OM模型并运行*main.py*调用*inference.py*完成模型训练与部署
