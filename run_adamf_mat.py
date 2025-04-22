from email.generator import Generator
import torch
import mmkgc
from mmkgc.config import Tester, AdvMixTrainer
from mmkgc.module.model import AdvMixRotatE
from mmkgc.module.loss import SigmoidLoss
from mmkgc.module.strategy import NegativeSampling
from mmkgc.data import TrainDataLoader, TestDataLoader
from mmkgc.adv.modules import MultiGenerator

from args import get_args

from mmkgc.module.qwen.Qwen import Qwen2_5_VL_Peft
from mmkgc.module.model import QwenAdvMixRotatE
import pickle

if __name__ == "__main__":
    args = get_args()
    print(args)
    # set the seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    qwen = Qwen2_5_VL_Peft()

    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/" + args.dataset + '/',
        batch_size=args.batch_size,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=args.neg_num,
        neg_rel=0
    )
    # dataloader for test
    test_dataloader = TestDataLoader(
        "./benchmarks/" + args.dataset + '/', "link")
    
    #TODO: to change img_emb, text_emb into Qwen2.5-VL
    
    #img_emb = torch.load('./embeddings/' + args.dataset + '-visual.pth')
    #text_emb = torch.load('./embeddings/' + args.dataset + '-textual.pth')
    img_emb = pickle.load(open('./embeddings/' + args.dataset + '-visual.pkl', 'rb'))
    text_emb = pickle.load(open('./embeddings/' + args.dataset + '-textual.pkl', 'rb'))
    # define the model
    kge_score = QwenAdvMixRotatE(
        qwen_model=args.qwen_model,
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.dim,
        margin=args.margin,
        epsilon=2.0,
        img_list=img_emb,
        text_list=text_emb
    )
    #kge_score = AdvMixRotatE(
    #    ent_tot=train_dataloader.get_ent_tot(),
    #    rel_tot=train_dataloader.get_rel_tot(),
    #    dim=args.dim,
    #    margin=args.margin,
    #    epsilon=2.0,
    #    img_emb=img_emb,
    #    text_emb=text_emb
    #)
    print(kge_score)
    # define the loss function
    model = NegativeSampling(
        model=kge_score,
        loss=SigmoidLoss(adv_temperature=args.adv_temp),
        batch_size=train_dataloader.get_batch_size(),
    )
    
    adv_generator = MultiGenerator(
        noise_dim=64,
        structure_dim=2*args.dim,
        img_dim=2*args.dim
    )
    # train the model
    trainer = AdvMixTrainer(
        model=model,
        data_loader=train_dataloader,
        train_times=args.epoch,
        alpha=args.learning_rate,
        use_gpu=True,
        opt_method='Adam',
        generator=adv_generator,
        lrg=args.lrg,
        mu=args.mu
    )

    trainer.run()
    kge_score.save_checkpoint(args.save)

    # test the model
    kge_score.load_checkpoint(args.save)
    tester = Tester(model=kge_score, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)
