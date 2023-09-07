

def predict(model, x):
    model.eval()
    pred = model(x).cpu()
    return pred

