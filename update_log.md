# 更新说明

### 2019.05.17
1. 在`run.py`中添加了`train_and_eval`功能，当`do_train`和`do_eval`都为`true`时使用，可设置参数`throttle_secs`调节`evaluate`的频率。

### 2019.05.18
在`Processor`中添加了`postprocess`功能，可在`predict`完成后对结果进行后处理，主要包括预测结果保存，`metrics`的计算和保存。
