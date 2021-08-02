import torch
import torchvision.models as models
import torch.autograd.profiler as profiler

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profiler.profile(record_shapes=True) as prof:
    with profiler.record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))


with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

with profiler.profile() as prof:
    with profiler.record_function("model_inference"):
        model(inputs)

prof.export_chrome_trace("trace.json")