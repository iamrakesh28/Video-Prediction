import numpy as np

def test_model(model, X, Y):
    #e1 = model.evaluate(X[700:800], Y[700:800], True)
    # test_loss = model.evaluate(X[800:], Y[800:], False)
    # print('Test Loss {:.4f}'.format(test_loss))

    y1 = model.predict(X[50], 10)
    # y2 = model.predict(X[915], 10)
    # y3 = model.predict(X[936], 10)
    # y4 = model.predict(X[956], 10)

    plot_result(
        np.squeeze(X[50].numpy(), axis=3),
        np.squeeze(Y[50].numpy(), axis=3),
        np.squeeze(y1, axis=3)
    )
        
    '''
    plot_result(
        X[915].numpy(),
        Y[915].numpy(),
        y2
    )
    
    plot_result(
        X[936].numpy(),
        Y[936].numpy(),
        y3
    )
    
    plot_result(
        X[956].numpy(),
        Y[956].numpy(),
        y4
    )
    '''
