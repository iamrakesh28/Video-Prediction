from utility import restore_patch, plot_result

def test_model(model, X, Y, batch_size):
    
    test_loss = model.evaluate(X[1000:1300], Y[1000:1300], batch_size)
    print('Test Loss {:.4f}'.format(test_loss))

    y1 = model.predict(X[50], 10)
    y2 = model.predict(X[1000], 10)
    y3 = model.predict(X[1500], 10)
    y4 = model.predict(X[1345], 10)

    plot_result(
        restore_patch(X[50].numpy(), (4, 4)),
        restore_patch(Y[50].numpy(), (4, 4)),
        restore_patch(y1, (4, 4))
    )
    
    plot_result(
        restore_patch(X[1000].numpy(), (4, 4)),
        restore_patch(Y[1000].numpy(), (4, 4)),
        restore_patch(y2, (4, 4))
    )
    
    plot_result(
        restore_patch(X[1500].numpy(), (4, 4)),
        restore_patch(Y[1500].numpy(), (4, 4)),
        restore_patch(y3, (4, 4))
    )
    
    plot_result(
        restore_patch(X[1345].numpy(), (4, 4)),
        restore_patch(Y[1345].numpy(), (4, 4)),
        restore_patch(y4, (4, 4))
    )
