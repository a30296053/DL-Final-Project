from DataSets import *
from MODEL  import *
from Config import  *
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
train_f1_list = []
val_f1_list =[]
torch.manual_seed(42)

def unison_shuffled(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def train(t_dataset,v_dataset ,model:nn.Module):
    train_loader = DataLoader(t_dataset,BATCH_SIZE,True)
    best_f1 = 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer =  torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=LAMBDA)
    criterion =  nn.BCEWithLogitsLoss()
    for epoch in range(EPOCH):
        model.train()
        running_loss,   n =  0.0, 0
        running_TP = 0
        running_FP = 0
        running_FN = 0
        
        for X, y in train_loader:
            X = X.to(device)
            y = y.unsqueeze(1).to(device)
            optimizer.zero_grad()
            logits =  model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            bs = y.size(0)
            running_loss +=  loss.item() * bs
            n += bs 

            prob =  torch.sigmoid(logits)
            pred  = (prob > 0.5).long()
            y_true =  y.long()
            running_TP += ((pred == 1) & (y_true == 1)).sum().item() 
            running_FP += ((pred == 1) & (y_true == 0)).sum().item()
            running_FN += ((pred == 0) & (y_true == 1)).sum().item()

        train_loss = running_loss / n
        precision = running_TP / (running_TP + running_FP + 1e-8)
        recall    = running_TP / (running_TP + running_FN + 1e-8)
        train_f1  = 2 * precision * recall / (precision + recall + 1e-8)
        val_loss, val_f1, _ = eval(v_dataset,model)
        train_f1_list.append(train_f1)
        val_f1_list.append(val_f1)
        print(f"Epoch {epoch:02d}/{EPOCH} | "
              f"Train Loss {train_loss:.4f} | Train F1 {train_f1:.4f} | "
              f"Val Loss {val_loss:.4f} | Val F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "./results/models/best_DRIAM.pt")
@torch.no_grad()
def eval(dataset,  model):
    data_loader = DataLoader(dataset)
    criterion =  nn.BCEWithLogitsLoss()
    loss_sum = 0
    n = 0
    TP = FP = FN =TN= 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    y_true_l = []
    y_pred_l = []
    for X, y in data_loader:
        X = X.to(device)
        y = y.unsqueeze(1).to(device)
        logits = model(X)
        loss = criterion(logits, y)
        bs = y.size(0)
        loss_sum += loss.item() * bs
        n +=  bs
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        y_true = y.long()
        y_true_l.append(y_true)
        y_pred_l.append(preds)
        TP += ((preds == 1) & (y_true == 1)).sum().item()
        TN += ((preds == 0) & (y_true == 0)).sum().item()
        FP += ((preds == 1) & (y_true == 0)).sum().item()
        FN += ((preds == 0) & (y_true == 1)).sum().item()

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return loss_sum / n, f1,(y_true_l,  y_pred_l)
    ...
def main():
    sizes = [INPUT_SIZE]
    i  = 4096
    while i >= 1:
        sizes.append(i)
        i//=2
    print(sizes)
    model = Funnel(sizes)

    X,Y = unison_shuffled(*get_data())
    n =  len(X)
    ts, vs = int(0.7*n),  int(0.15*n) 
    train_data = DRIAM(X[:ts], Y[:ts])
    valid_data  =  DRIAM(X[ts : ts+vs], Y[ts : ts+vs])
    test_data = DRIAM(X[ts+vs:],Y[ts+vs:])
    while(True):
        INPUT = input("Train y/n/q (y:train model, n:show result, q:quit) :")
        if INPUT == 'y':
            train(train_data, valid_data,model)
            epochs = range(1, EPOCH + 1)

            plt.figure(figsize=(8, 5))

            plt.plot(epochs, train_f1_list, color='red', label='Train F1')
            plt.plot(epochs, val_f1_list, color='blue', label='Validation F1')

            plt.xlabel("Epoch")
            plt.ylabel("F1 Score")
            plt.title("F1 Score vs Epoch")
            plt.legend()
            plt.grid(True)

            plt.show()
        elif INPUT == 'n':
            try:
                model.load_state_dict(torch.load("./results/models/best_DRIAM.pt"))
                loss,  f1 ,confu= eval(test_data,model)
                print(  f"Test Loss | {loss:.4f} "
                        f"Test F1 | {f1:.4f}")
                y_true = torch.tensor(confu[0])
                y_pred =  torch.tensor(confu[1])
                cm = confusion_matrix(y_true,y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])        
                disp.plot(cmap='Blues')
                plt.show()
            except Exception as E:
                print(E)
        else :
            break
        

    
    ...
if __name__ == '__main__' :
    main()