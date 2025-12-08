from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch 

class Trainer(BaseTrainer):
    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):

        if self.writer is None:
            return

        if mode == "train":
            if batch_idx % self.log_step != 0:
                return

        sr = 22050
        try:
            if "mel_config" in self.config.model:
                sr = int(self.config.model.mel_config.sr)
        except Exception:
            pass

        if "audio_hat" in batch:
            audio_hat = batch["audio_hat"].detach().cpu()
            if audio_hat.dim() == 3:
                audio_hat = audio_hat[0, 0] 
            elif audio_hat.dim() == 2:
                audio_hat = audio_hat[0]
            self.writer.add_audio(f"{mode}_audio_hat", audio_hat, sample_rate=sr)

        if "audio" in batch:
            audio = batch["audio"].detach().cpu()
            if audio.dim() == 3:
                audio = audio[0, 0]
            elif audio.dim() == 2:
                audio = audio[0]
            self.writer.add_audio(f"{mode}_audio_gt", audio, sample_rate=sr)

        if "mel" in batch:
            mel = batch["mel"].detach().cpu()
            # [B, 80, T] -> [80, T]
            if mel.dim() == 3:
                mel = mel[0]
            elif mel.dim() != 2:
                return

            mel_np = mel.numpy()
            self.writer.add_image(f"{mode}_mel", mel_np)
