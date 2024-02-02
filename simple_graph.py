import torch

# def forward(inputs):
#     # No stacktrace found for following nodes
#     getitem: "f32[]" = inputs[0]
#     getitem_1: "f32[1, 2]" = inputs[1]
#     getitem_2: "f32[2, 2]" = inputs[2];  inputs = None
#     expand: "f32[1, 2]" = torch.ops.aten.expand.default(getitem, [1, 2]);  getitem = None
#     t: "f32[2, 1]" = torch.ops.aten.t.default(expand);  expand = None
#     mm: "f32[2, 2]" = torch.ops.aten.mm.default(t, getitem_1);  t = getitem_1 = None

#     # manually edited:
#     accumulate_grad_ = torch.ops.inductor.accumulate_grad_.default(getitem_2, mm);  getitem_2 = t_2 = None # grad is copied instead of moved
#     # t_1: "f32[2, 2]" = torch.ops.aten.t.default(mm);  mm = None
#     # t_2: "f32[2, 2]" = torch.ops.aten.t.default(t_1);  t_1 = None
#     # accumulate_grad_ = torch.ops.inductor.accumulate_grad_.default(getitem_2, t_2);  getitem_2 = t_2 = None
#     return []

# x = torch.randn([1, 2])
# linear = torch.nn.Linear(2, 2, bias=False)
# inputs = [x, x, linear.weight]
# forward(inputs)



accumulate_grad_ = torch.ops.inductor.accumulate_grad_.default(
    torch.randn([2,2]),
    [torch.randn([2,2])]
)
