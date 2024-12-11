import math
import torch.nn as nn
import torch
from tqdm import tqdm
import os
from torch import optim
import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer, AutoTokenizer
from torch.utils.data import DataLoader

from src.utils import *
from src.modules import *

class ModelBart(nn.Module):
    def __init__(
            self,
            model_name_or_path: str = 'facebook/mbart-large-50-many-to-many-mmt',
            local_files_only = True,
            src_lang = "zh_CN",
            tgt_lang = "en_XX",
            device: str = "cpu",
        ):
        super().__init__()
        if model_name_or_path == None:
            self.tokenizer = None
            self.model = None
        else:
            self.tokenizer = MBartTokenizer.from_pretrained(model_name_or_path,local_files_only = local_files_only)
            self.model = MBartForConditionalGeneration.from_pretrained(model_name_or_path, local_files_only = local_files_only)
            self.tokenizer.src_lang = src_lang
            self.tokenizer.tgt_lang = tgt_lang
        
        self.device = device if torch.cuda.is_available() else "cpu"
    
    def forward(self,batch_data_encoder,batch_data_decoder):
        output = self.model(batch_data_encoder['input_ids'].to(self.device),batch_data_encoder['attention_mask'].to(self.device),decoder_input_ids = batch_data_decoder["input_ids"],decoder_attention_mask = batch_data_decoder["attention_mask"])
        return output

    def load_pretrained(self,save_model_dir:str):
        if os.path.isdir(save_model_dir) and os.listdir(save_model_dir):
            self.tokenizer = MBartTokenizer.from_pretrained(save_model_dir, local_files_only=True)
            self.model = MBartForConditionalGeneration.from_pretrained(save_model_dir, local_files_only=True)
        self.model = self.model.to(self.device)

    def save_pretrained(self,save_model_dir:str):
        self.model.save_pretrained(save_model_dir)
        self.tokenizer.save_pretrained(save_model_dir)
    
class ModelPretrainForTranslateBaseBart(nn.Module):
    def __init__(
        self, 
        pretrain_model_name_or_path = "facebook/mbart-large-50-many-to-many-mmt",
        device = "cpu"
    ):
        super().__init__()
        self.model_name_or_path = pretrain_model_name_or_path
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model = ModelBart(
            model_name_or_path = self.model_name_or_path,
            device = self.device
        )
        self.tokenizer = self.model.tokenizer

    def forward(self,input:dict):
        batch_data = {
            "encoder_seq":{
                "input_ids":input["x_encoder"]["input_ids"].reshape(-1,input["x_encoder"]["input_ids"].shape[2]).to(self.device),
                "attention_mask":input["x_encoder"]["attention_mask"].reshape(-1,input["x_encoder"]["attention_mask"].shape[2]).to(self.device)
            },
            "decoder_seq":{
                "input_ids":input["x_decoder"]["input_ids"].reshape(-1,input["x_decoder"]["input_ids"].shape[2]).to(self.device),
                "attention_mask":input["x_decoder"]["attention_mask"].reshape(-1,input["x_decoder"]["attention_mask"].shape[2]).to(self.device)
            }
        }
        outputs =  self.model(batch_data["encoder_seq"],batch_data["decoder_seq"])

        mask = batch_data["decoder_seq"]["attention_mask"][..., 1:]
        labels =  batch_data["decoder_seq"]["input_ids"][..., 1:]
        predict = outputs["logits"][..., :-1, :]
        B, T, C = predict.shape
        predict = predict.reshape(B*T, C)
        labels = labels.reshape(-1)
        mask = mask.reshape(-1)

        output = {
            "predict":predict,
            "label":labels,
            "mask":mask
        }

        return output

    def load_pretrained(self, save_model_dir):
        self.model.load_pretrained(save_model_dir)

    def save_pretrained(self,  save_model_dir):
        self.model.save_pretrained(save_model_dir)
        
    def compute_loss(self, input:dict):
        output = {}
        if "class_weights" not in input:
            input["class_weights"] = None
        if "mask" not in input:
            self.criterion = nn.CrossEntropyLoss(weight = input["class_weights"])
            output["total_loss"] = self.criterion(input["predict"],input["label"])
        else:
            self.criterion = nn.CrossEntropyLoss(weight = input["class_weights"], reduction = 'none')
            loss = self.criterion(input["predict"],input["label"])
            masked_loss = loss * input["mask"]
            output["total_loss"] = masked_loss.sum() / input["mask"].sum()
        return output

    def train_one_epoch(self, epoch,train_dataloader, optimizer, clip_max_norm, log_path = None):
        self.train()
        self.to(self.device)
        pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
        total_loss = AverageMeter()
        average_hit_rate = AverageMeter()
        for batch_id, inputs in enumerate(train_dataloader):
            """ grad zeroing """
            optimizer.zero_grad()

            """ forward """
            used_memory = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()
            total_loss.update(out_criterion["total_loss"].item())
            average_hit_rate.update(math.exp(-total_loss.avg))

            """ grad clip """
            if clip_max_norm > 0:
                clip_gradient(optimizer,clip_max_norm)

            """ modify parameters """
            optimizer.step()
            after_used_memory = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            postfix_str = "total_loss: {:.4f},average_hit_rate:{:.4f},use_memory: {:.1f}G".format(
                total_loss.avg, 
                average_hit_rate.avg,
                after_used_memory - used_memory
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
        with open(log_path, "a") as file:
            file.write(postfix_str+"\n")
        return total_loss.avg

    def test_epoch(self,epoch, test_dataloader,trainning_log_path = None):
        total_loss = AverageMeter()
        average_hit_rate = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for batch_id, inputs in enumerate(test_dataloader):
                """ forward """
                output = self.forward(inputs)

                """ calculate loss """
                out_criterion = self.compute_loss(output)
                total_loss.update(out_criterion["total_loss"])

            average_hit_rate.update(math.exp(-total_loss.avg))
            str = "Test Epoch: {:d}, total_loss: {:.4f},average_hit_rate:{:.4f}".format(
                epoch,
                total_loss.avg, 
                average_hit_rate.avg,
            )
        print(str)
        with open(trainning_log_path, "a") as file:
            file.write(str+"\n")
        return total_loss.avg

    def trainning(
        self,
        train_dataloader:DataLoader = None,
        test_dataloader:DataLoader = None,
        optimizer_name:str = "Adam",
        weight_decay:float = 1e-4,
        clip_max_norm:float = 0.5,
        factor:float = 0.3,
        patience:int = 15,
        lr:float = 1e-4,
        total_epoch:int = 1000,
        save_checkpoint_step:str = 10,
        save_model_dir:str = "models"
    ):
        ## 1 trainning log path 
        first_trainning = True
        check_point_path = save_model_dir  + "/checkpoint.pth"
        log_path = save_model_dir + "/train.log"

        ## 2 get net pretrain parameters if need 
        """
            If there is  training history record, load pretrain parameters
        """
        if  os.path.isdir(save_model_dir) and os.path.exists(check_point_path) and os.path.exists(log_path):
            self.load_pretrained(save_model_dir)  
            first_trainning = False

        else:
            if not os.path.isdir(save_model_dir):
                os.makedirs(save_model_dir)
            with open(log_path, "w") as file:
                pass


        ##  3 get optimizer
        if optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(self.parameters(),lr,weight_decay = weight_decay)
        else:
            optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer, 
            mode = "min", 
            factor = factor, 
            patience = patience
        )

        ## init trainng log
        if first_trainning:
            best_loss = float("inf")
            last_epoch = 0
        else:
            checkpoint = torch.load(check_point_path, map_location=self.device)
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            best_loss = checkpoint["loss"]
            last_epoch = checkpoint["epoch"] + 1

        try:
            for epoch in range(last_epoch,total_epoch):
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                train_loss = self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm,log_path)
                test_loss = self.test_epoch(epoch,test_dataloader,log_path)
                loss = train_loss + test_loss
                lr_scheduler.step(loss)
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                check_point_path = save_model_dir  + "/checkpoint.pth"
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": None,
                        "lr_scheduler": None
                    },
                    check_point_path
                )

                if epoch % save_checkpoint_step == 0:
                    os.makedirs(save_model_dir + "/" + "chaeckpoint-"+str(epoch))
                    torch.save(
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict()
                        },
                        save_model_dir + "/" + "chaeckpoint-"+str(epoch)+"/checkpoint.pth"
                    )
                if is_best:
                    self.save_pretrained(save_model_dir)

        # interrupt trianning
        except KeyboardInterrupt:
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict()
                    },
                    check_point_path
                )
                
class ModelForTranslate(nn.Module):
    def __init__(
        self, 
        tokenizer_name_or_path = "BAAI/bge-reranker-base",
        vecab_dim = 1024, 
        n_head = 16,
        ffn_hidden = 512, 
        n_layers = 6, 
        drop_prob = 0.1, 
        max_seq_len = 1024,
        device = "cpu"
    ):
        super().__init__()
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path,local_files_only=True)
        self.vocab_size = self.tokenizer.vocab_size
        self.vecab_dim = vecab_dim
        self.cross = nn.CrossEntropyLoss(reduction='none')
        self.embedding = nn.Embedding(self.vocab_size, self.vecab_dim, dtype=torch.float32,device = self.device)

        self.backone = ModelTransformer(
            d_model_encoder = self.vecab_dim, 
            d_model_decoder = self.vecab_dim,
            n_head = n_head,
            ffn_hidden =ffn_hidden, 
            n_layers = n_layers, 
            drop_prob = drop_prob, 
            max_seq_len = max_seq_len, 
            device = device
        )

        self.predict_head = nn.Sequential(
            nn.Linear(self.vecab_dim, self.vocab_size)
        ).to(device)

    def forward(self,input):
        output = {}
        batch_data = {
            "encoder_seq":{
                "input_ids":input["x_encoder"]["input_ids"].reshape(-1,input["x_encoder"]["input_ids"].shape[2]).to(self.device),
                "attention_mask":input["x_encoder"]["attention_mask"].reshape(-1,input["x_encoder"]["attention_mask"].shape[2]).to(self.device)
            },
            "decoder_seq":{
                "input_ids":input["x_decoder"]["input_ids"].reshape(-1,input["x_decoder"]["input_ids"].shape[2]).to(self.device),
                "attention_mask":input["x_decoder"]["attention_mask"].reshape(-1,input["x_decoder"]["attention_mask"].shape[2]).to(self.device)
            }
        }
        encoder_seq = batch_data["encoder_seq"]
        decoder_seq = batch_data["decoder_seq"]
        encoder_seq["input"] = self.embedding(encoder_seq["input_ids"])
        decoder_seq["input"] = self.embedding(decoder_seq["input_ids"])
        y = self.backone(encoder_seq,decoder_seq,pos_emb = "absolute")
        logits = self.predict_head(y)
        mask = batch_data["decoder_seq"]["attention_mask"][..., 1:]
        labels =  batch_data["decoder_seq"]["input_ids"][..., 1:]
        predict = logits[..., :-1, :]
        B, T, C = predict.shape
        predict = predict.reshape(B*T, C)
        labels = labels.reshape(-1)
        mask = mask.reshape(-1)
        output = {
            "predict":predict,
            "label":labels,
            "mask":mask
        }
        return output

    def load_pretrained(self,save_model_dir):
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"))

    def save_pretrained(self,save_model_dir):
        torch.save(self.state_dict(), save_model_dir + "/model.pth")

        
    def compute_loss(self, input:dict):
        output = {}
        if "class_weights" not in input:
            input["class_weights"] = None
        if "mask" not in input:
            self.criterion = nn.CrossEntropyLoss(weight = input["class_weights"])
            output["total_loss"] = self.criterion(input["predict"],input["label"])
        else:
            self.criterion = nn.CrossEntropyLoss(weight = input["class_weights"], reduction = 'none')
            loss = self.criterion(input["predict"],input["label"])
            masked_loss = loss * input["mask"]
            output["total_loss"] = masked_loss.sum() / input["mask"].sum()
        return output

    def train_one_epoch(self, epoch,train_dataloader, optimizer, clip_max_norm, log_path = None):
        self.train()
        self.to(self.device)
        pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
        total_loss = AverageMeter()
        average_hit_rate = AverageMeter()
        for batch_id, inputs in enumerate(train_dataloader):
            """ grad zeroing """
            optimizer.zero_grad()

            """ forward """
            used_memory = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()
            total_loss.update(out_criterion["total_loss"].item())
            average_hit_rate.update(math.exp(-total_loss.avg))

            """ grad clip """
            if clip_max_norm > 0:
                clip_gradient(optimizer,clip_max_norm)

            """ modify parameters """
            optimizer.step()
            after_used_memory = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            postfix_str = "total_loss: {:.4f},average_hit_rate:{:.4f},use_memory: {:.1f}G".format(
                total_loss.avg, 
                average_hit_rate.avg,
                after_used_memory - used_memory
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
        with open(log_path, "a") as file:
            file.write(postfix_str+"\n")
        return total_loss.avg

    def test_epoch(self,epoch, test_dataloader,trainning_log_path = None):
        total_loss = AverageMeter()
        average_hit_rate = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for batch_id, inputs in enumerate(test_dataloader):
                """ forward """
                output = self.forward(inputs)

                """ calculate loss """
                out_criterion = self.compute_loss(output)
                total_loss.update(out_criterion["total_loss"])

            average_hit_rate.update(math.exp(-total_loss.avg))
            str = "Test Epoch: {:d}, total_loss: {:.4f},average_hit_rate:{:.4f}".format(
                epoch,
                total_loss.avg, 
                average_hit_rate.avg,
            )
        print(str)
        with open(trainning_log_path, "a") as file:
            file.write(str+"\n")
        return total_loss.avg

    def trainning(
        self,
        train_dataloader:DataLoader = None,
        test_dataloader:DataLoader = None,
        optimizer_name:str = "Adam",
        weight_decay:float = 1e-4,
        clip_max_norm:float = 0.5,
        factor:float = 0.3,
        patience:int = 15,
        lr:float = 1e-4,
        total_epoch:int = 1000,
        save_checkpoint_step:str = 10,
        save_model_dir:str = "models"
    ):
        ## 1 trainning log path 
        first_trainning = True
        check_point_path = save_model_dir  + "/checkpoint.pth"
        log_path = save_model_dir + "/train.log"

        ## 2 get net pretrain parameters if need 
        """
            If there is  training history record, load pretrain parameters
        """
        if  os.path.isdir(save_model_dir) and os.path.exists(check_point_path) and os.path.exists(log_path):
            self.load_pretrained(save_model_dir)  
            first_trainning = False

        else:
            if not os.path.isdir(save_model_dir):
                os.makedirs(save_model_dir)
            with open(log_path, "w") as file:
                pass


        ##  3 get optimizer
        if optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(self.parameters(),lr,weight_decay = weight_decay)
        else:
            optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer, 
            mode = "min", 
            factor = factor, 
            patience = patience
        )

        ## init trainng log
        if first_trainning:
            best_loss = float("inf")
            last_epoch = 0
        else:
            checkpoint = torch.load(check_point_path, map_location=self.device)
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            best_loss = checkpoint["loss"]
            last_epoch = checkpoint["epoch"] + 1

        try:
            for epoch in range(last_epoch,total_epoch):
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                train_loss = self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm,log_path)
                test_loss = self.test_epoch(epoch,test_dataloader,log_path)
                loss = train_loss + test_loss
                lr_scheduler.step(loss)
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                check_point_path = save_model_dir  + "/checkpoint.pth"
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": None,
                        "lr_scheduler": None
                    },
                    check_point_path
                )

                if epoch % save_checkpoint_step == 0:
                    os.makedirs(save_model_dir + "/" + "chaeckpoint-"+str(epoch))
                    torch.save(
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict()
                        },
                        save_model_dir + "/" + "chaeckpoint-"+str(epoch)+"/checkpoint.pth"
                    )
                if is_best:
                    self.save_pretrained(save_model_dir)

        # interrupt trianning
        except KeyboardInterrupt:
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict()
                    },
                    check_point_path
                )
                