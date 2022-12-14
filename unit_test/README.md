
IPCAvsLMIPCA.py: Test the two implementations of the same layer with same input. If the test passes
they should output the same.

```python
python IPCAvsLMIPCA.py

Output from original IPCA
tensor([[[-0.1077, -0.0912, -0.0562,  ..., -0.1033,  0.1245,  0.2865],
         [-0.1094, -0.0763, -0.0503,  ..., -0.1108,  0.1063,  0.2816],
         [-0.0981, -0.0843, -0.0585,  ..., -0.1037,  0.1161,  0.2811],
         ...,
         [-0.1076, -0.0802, -0.0562,  ..., -0.0972,  0.1244,  0.2913],
         [-0.1035, -0.0892, -0.0643,  ..., -0.1153,  0.1109,  0.2809],
         [-0.1043, -0.0743, -0.0568,  ..., -0.1010,  0.1181,  0.2886]]],
       grad_fn=<AddBackward0>) torch.Size([1, 256, 768])

Output from linear memory IPCA
tensor([[[-0.1077, -0.0912, -0.0562,  ..., -0.1033,  0.1245,  0.2865],
         [-0.1094, -0.0763, -0.0503,  ..., -0.1108,  0.1063,  0.2816],
         [-0.0981, -0.0843, -0.0585,  ..., -0.1037,  0.1161,  0.2811],
         ...,
         [-0.1076, -0.0802, -0.0562,  ..., -0.0972,  0.1244,  0.2913],
         [-0.1035, -0.0892, -0.0643,  ..., -0.1153,  0.1109,  0.2809],
         [-0.1043, -0.0743, -0.0568,  ..., -0.1010,  0.1181,  0.2886]]],
       grad_fn=<AddBackward0>) torch.Size([1, 256, 768])
```


ATTNvsLMATTN.py: Test the two implementationso of the BertSelfAttention layer with same input,
setting no dropout prob for consistency. They should be the same if the test passes.

```
python ATTNvsLMATTN.py

Output from original attention
(tensor([[[-0.0304,  0.0459, -0.2800,  ...,  0.0776, -0.0980, -0.6149],
         [-0.0310,  0.0453, -0.2807,  ...,  0.0777, -0.0980, -0.6151],
         [-0.0305,  0.0448, -0.2808,  ...,  0.0781, -0.0976, -0.6154],
         ...,
         [-0.0309,  0.0459, -0.2802,  ...,  0.0773, -0.0980, -0.6152],
         [-0.0310,  0.0451, -0.2805,  ...,  0.0781, -0.0981, -0.6150],
         [-0.0310,  0.0447, -0.2805,  ...,  0.0781, -0.0978, -0.6150]]],
       grad_fn=<ViewBackward0>),) torch.Size([1, 256, 768])

Output from linear memory attention
(tensor([[[-0.0304,  0.0459, -0.2800,  ...,  0.0776, -0.0980, -0.6149],
         [-0.0310,  0.0453, -0.2807,  ...,  0.0777, -0.0980, -0.6151],
         [-0.0305,  0.0448, -0.2808,  ...,  0.0781, -0.0976, -0.6154],
         ...,
         [-0.0309,  0.0459, -0.2802,  ...,  0.0773, -0.0980, -0.6152],
         [-0.0310,  0.0451, -0.2805,  ...,  0.0781, -0.0981, -0.6150],
         [-0.0310,  0.0447, -0.2805,  ...,  0.0781, -0.0978, -0.6150]]],
       grad_fn=<ViewBackward0>),) torch.Size([1, 256, 768])
```
