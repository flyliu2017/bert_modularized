# 更新说明

### 2019.05.17
1. 在`run.py`中添加了`train_and_eval`功能，当`do_train`和`do_eval`都为`true`时使用，可设置参数`throttle_secs`调节`evaluate`的频率。

### 2019.05.18
在`Processor`中添加了`postprocess`功能，可在`predict`完成后对结果进行后处理，主要包括预测结果保存，`metrics`的计算和保存。

### 2019.05.22
添加了`predict_from_file`功能,设置`do_predict=true`并将`predict_from_file`设为文件路径，可以将该文件作为输入进行预测。  
修复了`post_process`中的bug。

### 2019.05.23
在`optimization`中添加了新的衰减方式`exponential decay`。