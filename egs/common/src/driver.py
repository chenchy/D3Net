import musdb
import museval

import torch
import torch.nn as nn

MIN_PESQ = -0.5

class TrainerBase:
    def __init__(self, model, loader, criterion, optimizer, args):
        self.train_loader, self.valid_loader = loader['train'], loader['valid']
        
        self.model = model
        
        self.criterion = criterion
        self.optimizer = optimizer
        
        self._reset(args)
    
    def _reset(self, args):
        raise NotImplementedError("Implement `_reset` in the sub-class.")
    
    def run(self):
        raise NotImplementedError("Implement `run` in the sub-class.")
        
    def run_one_epoch(self, epoch):
        """
        Training
        """
        train_loss = self.run_one_epoch_train(epoch)
        valid_loss = self.run_one_epoch_eval(epoch)

        return train_loss, valid_loss
    
    def run_one_epoch_train(self, epoch):
        raise NotImplementedError("Implement `run_one_epoch_train` in the sub-class.")
    
    def run_one_epoch_eval(self, epoch):
        raise NotImplementedError("Implement `run_one_epoch_eval` in the sub-class.")
    
    def save_model(self, epoch, model_path='./tmp.pth'):
        if isinstance(self.model, nn.DataParallel):
            package = self.model.module.get_package()
            package['state_dict'] = self.model.module.state_dict()
        else:
            package = self.model.get_package()
            package['state_dict'] = self.model.state_dict()
            
        package['optim_dict'] = self.optimizer.state_dict()
        
        package['best_loss'] = self.best_loss
        package['no_improvement'] = self.no_improvement
        
        package['train_loss'] = self.train_loss
        package['valid_loss'] = self.valid_loss
        
        package['epoch'] = epoch + 1
        
        torch.save(package, model_path)

class TesterBase:
    def __init__(self, model, loader, criterion, args):
        self.loader = loader
        
        self.model = model
        
        self.criterion = criterion
        
        self._reset(args)
        
    def _reset(self, args):
        raise NotImplementedError("Implement `_reset` in the sub-class.")

    def run(self):
        raise NotImplementedError("Implement `run` in the sub-class.")

class EvaluaterBase:
    def __init__(self, args):
        self._reset(args)
    
    def _reset(self, args):
        self.target = [
            'drums', 'bass', 'other', 'vocals', 'accompaniment'
        ]
        self.mus = musdb.DB(root=args.musdb18_root, subsets="test", is_wav=True)
        self.estimated_mus = musdb.DB(root=args.estimated_musdb18_root, subsets="test", is_wav=True)
        self.json_dir = args.json_dir
    
    def run(self):
        results = museval.EvalStore(frames_agg='median', tracks_agg='median')

        for track, estimated_track in zip(self.mus, self.estimated_mus):
            scores = self.run_one_track(track, estimated_track)
            results.add_track(scores)
        
        print(results)
    
    def run_one_track(self, track, estimated_track):
        estimates = {}

        for _target in self.target:
            estimates[_target] = estimated_track.targets[_target].audio
        
        scores = museval.eval_mus_track(
            track, estimates, output_dir=self.json_dir
        )

        return scores