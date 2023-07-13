# [YOLOv8 pruning](benchmarks\prunability\yolov8_pruning.py)

## 1. [tp.pruner.MagnitudePruner](torch_pruning/pruner/algorithms/metapruner.py)
    
Đối tượng này dùng để khởi tạo một pruner cho một model

```python
class MetaPruner:
    def __init__(
        self,
        # Basic
        model: nn.Module,
        example_inputs: torch.Tensor,
        importance: typing.Callable,
        # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        global_pruning: bool = False,
        ch_sparsity: float = 0.5,  # channel/dim sparsity
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None,
        max_ch_sparsity: float = 1.0,
        iterative_steps: int = 1,  # for iterative pruning
        iterative_sparsity_scheduler: typing.Callable = linear_scheduler,
        ignored_layers: typing.List[nn.Module] = None,

        # Advanced
        round_to: int = None,  # round channels to 8x, 16x, ...
        # for grouped channels.
        channel_groups: typing.Dict[nn.Module, int] = dict(),
        # pruners for customized layers
        customized_pruners: typing.Dict[typing.Any,
                                        function.BasePruningFunc] = None,
        # unwrapped nn.Parameters like ViT.pos_emb
        unwrapped_parameters: typing.List[nn.Parameter] = None,
        root_module_types: typing.List = [
            ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        output_transform: typing.Callable = None,
    ):
```

### Important params
    - model (nn.Module): model cần prune
    - example_inputs (torch.Tensor or List): input cho model để xác định dependency graph
    - importance (Callable): (vd: tp.importance.MagnitudeImportance(p=2)) tiêu chí pruning
    - iterative_steps : số lần lặp prune
    - iterative_sparsity_scheduler (Callable): scheduler for iterative pruning. (vd: function tính toán tỷ lệ pruning cho mỗi epoch)
    - ch_sparsity: tỷ lệ prune
    - ignored_layers (List[nn.Module]): những layer bỏ qua không prune
    - unwrapped_parameters (list): nn.Parameter that does not belong to any supported layerss.

### Pruning
Sau khi khai báo một pruner, để pruning, tiến hành tạo một loop và gọi ```pruner.step()``` trong mỗi group để tiến hành prune với tỷ lệ đã tạo  trong ```iterative_sparsity_scheduler``` 

```python
model = resnet18(pretrained=True)
example_inputs = torch.randn(1, 3, 224, 224)

# 0. importance criterion for parameter selections
imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')

# 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m) # DO NOT prune the final classifier!

        
# 2. Pruner initialization
iterative_steps = 5 # You can prune your model to the target sparsity iteratively.
pruner = tp.pruner.MagnitudePruner(
    model, 
    example_inputs, 
    global_pruning=False, # If False, a uniform sparsity will be assigned to different layers.
    importance=imp, # importance criterion for parameter selection
    iterative_steps=iterative_steps, # the number of iterations to achieve target sparsity
    ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers,
)

base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
for i in range(iterative_steps):
    # 3. the pruner.step will remove some channels from the model with least importance
    pruner.step()
    
    # 4. Do whatever you like here, such as fintuning
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(model)
    print(model(example_inputs).shape)
    print(
        "  Iter %d/%d, Params: %.2f M => %.2f M"
        % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
    )
    print(
        "  Iter %d/%d, MACs: %.2f G => %.2f G"
        % (i+1, iterative_steps, base_macs / 1e9, macs / 1e9)
    )
    # finetune your model here
    # finetune(model)
    # ...

```

## 2. YOLOv8 prune
### 2.1 Quá trình train của yolov8
    Khởi tạo model: YOLO("model.pt")
    Bắt đầu train: model.train(...)
Khi gọi model.train của đối tượng YOLO, một chuỗi các func được gọi đến nhau, hàm thực hiện training cho mô hình sau chuỗi func được gọi là ```_do_train``` trong [trainer.py](ultralytics\yolo\engine\trainer.py). 

### 2.2 Vấn đề khi prune yolov8
Khi đưa model v8 vào ```pruner``` và gọi ```pruner.step()```, lỗi sẽ xảy ra ở module ```C2f```. Để giải quyết, ```C2f_v2``` được tạo ra để thay thế ```C2f``` cũ trong model. 

Với mỗi vòng lặp, model mà đối tượng YOLO refer đến sẽ bị thay đổi, do đó khi đưa model vào  ```pruner``` và gọi ```pruner.step()``` kết hợp với training model nhiều lần, model sẽ bị thay đổi reference và model trong ```pruner``` là một model cũ, model mới sẽ không được prune.

Sau khi train, model được lưu từ dạng FP32 về FP16. Trong quá trình prune + train, việc mapping liên tục giữa 2 miền biểu diễn làm cho mô hình giảm độ chính xác. 

### 2.3 Quá trình pruning yolov8
Tạo ra các module, function: ```C2f_v2, transfer_weights, replace_c2f_with_c2f_v2, save_model_v2, final_eval_v2, strip_optimizer_v2, train_v2``` để modify cấu trúc và quá trình training, validation của yolov8. 

Quá trình prune

```python
def prune(args):
    # load trained yolov8 model
    model = YOLO(args.model) # YOLO object
    model.__setattr__("train_v2", train_v2.__get__(model)) #add train_v2 function to model
    pruning_cfg = yaml_load(check_yaml(args.cfg))
    batch_size = pruning_cfg['batch'] = args.batch

    # modify gpu
    pruning_cfg['device'] = args.device

    #modify workers
    pruning_cfg['workers'] = args.workers

    # save results in folder
    pruning_cfg["project"] = args.project
    prefix_folder = f'train{args.no_runs}'

    # use coco128 dataset for 10 epochs fine-tuning each pruning iteration step
    # this part is only for sample code, number of epochs should be included in config file
    pruning_cfg['data'] = args.data
    pruning_cfg['epochs'] = args.epochs
    pruning_cfg["imgsz"] = args.imgsz

    model.model.train()

    replace_c2f_with_c2f_v2(model.model)
    initialize_weights(model.model)  # set BN.eps, momentum, ReLU.inplace

    for name, param in model.model.named_parameters():
        param.requires_grad = True

    example_inputs = torch.randn(1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"]).to(model.device)
    macs_list, nparams_list, map_list, pruned_map_list = [], [], [], []
    base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)

    # do validation before pruning model
    pruning_cfg['name'] = os.path.join(prefix_folder, f"baseline_val")
    pruning_cfg['batch'] = 1
    validation_model = deepcopy(model) #YOLO object
    metric = validation_model.val(**pruning_cfg)
    init_map = metric.box.map
    macs_list.append(base_macs)
    nparams_list.append(100)
    map_list.append(init_map)
    pruned_map_list.append(init_map)
    print(f"Before Pruning: MACs={base_macs / 1e9: .5f} G, #Params={base_nparams / 1e6: .5f} M, mAP={init_map: .5f}")

    # prune same ratio of filter based on initial size
    # áp dụng bài toán lãi suất kép để tính ch_sparsity cho mỗi iterative_steps
    ch_sparsity = 1 - math.pow((1 - args.target_prune_rate), 1 / args.iterative_steps)
    '''
        giải thích:
            - Giả sử số lượng chanel ban đầu là Ao, số lượng chanel sau prune là An = A0 * (1-p) 
              với p = args.target_prune_rate, n = args.iterative_steps
            -> cần tìm c = ch_sparsity cho mỗi lần lặp pruning
            - Lặp:
                + A1 = A0(1 - c)
                + A2 = A1(1 - c) = A0(1-c)^2
                ...
                + An = A0(1 - c)^n 

            mà An = A0 * (1 - p)
            => (1 - c)^n = (1 - p)
            => c = 1 - (1-p) ^ (1/n) (dpcm)
    '''
    
    for i in range(args.iterative_steps):

        model.model.train()
        for name, param in model.model.named_parameters():
            param.requires_grad = True

        ignored_layers = []
        unwrapped_parameters = []
        for m in model.model.modules():
            if isinstance(m, (Detect,)):
                ignored_layers.append(m)
        
        example_inputs = example_inputs.to(model.device)
        pruner = tp.pruner.MagnitudePruner(
            model.model,
            example_inputs,
            importance=tp.importance.MagnitudeImportance(p=2),  # L2 norm pruning,
            iterative_steps=1,
            ch_sparsity=ch_sparsity,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters
        )
        pruner.step() # remove some weights with lowest importance

        # pre fine-tuning validation
        pruning_cfg['name'] = os.path.join(prefix_folder, f"step_{i}_pre_val")
        pruning_cfg['batch'] = 1
        validation_model.model = deepcopy(model.model)
        metric = validation_model.val(**pruning_cfg)
        pruned_map = metric.box.map
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs)
        current_speed_up = float(macs_list[0]) / pruned_macs
        print(f"After pruning iter {i + 1}: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, "
              f"mAP={pruned_map}, speed up={current_speed_up}")

        # fine-tuning
        for name, param in model.model.named_parameters():
            param.requires_grad = True
        pruning_cfg['name'] = os.path.join(prefix_folder, f"step_{i}_finetune")
        pruning_cfg['batch'] = batch_size  # restore batch size
        model.train_v2(pruning=True, **pruning_cfg)

        # post fine-tuning validation
        pruning_cfg['name'] = os.path.join(prefix_folder, f"step_{i}_post_val")
        pruning_cfg['batch'] = 1
        validation_model = YOLO(model.trainer.best)
        metric = validation_model.val(**pruning_cfg)
        current_map = metric.box.map
        print(f"After fine tuning mAP={current_map}")

        macs_list.append(pruned_macs)
        nparams_list.append(pruned_nparams / base_nparams * 100)
        pruned_map_list.append(pruned_map)
        map_list.append(current_map)

        # remove pruner after single iteration
        del pruner

        save_pruning_performance_graph(nparams_list, map_list, macs_list, pruned_map_list, 
                                       dir = os.path.join(pruning_cfg["project"], prefix_folder) )

        if init_map - current_map > args.max_map_drop:
            print("Pruning early stop")
            break
    model.export(format='onnx')
    model.export(format='torchscript')
```

Do mỗi lần training, model của đối tượng YOLO được reference đến sẽ bị thay đổi, giải pháp được dưa ra là initialize ```pruner``` repeatedly để refer đến model của đối tượng liên tục. Trong quá trình pruning, model được training, evaluate để khôi phục lại mAPs đã mất do prune.

Việc initialize ```pruner``` liên tục khiến cho việc tính tỷ lệ prune ở mỗi bước trở nên phức tạp hơn do ```ch_sparsity``` cần set cho pruner của model đang prune ở bước lặp hiện tại, không phải model full ban đầu. 