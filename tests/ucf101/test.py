from utility import restore_patch, plot_result

def test_model(model, X, Y, batch_size):
    
    test_loss = model.evaluate(X[300:], Y[300:], batch_size)
    print('Test Loss {:.4f}'.format(test_loss))

    y1 = model.predict(X[50], 5)
    y2 = model.predict(X[365], 5)
    y3 = model.predict(X[410], 5)
    y4 = model.predict(X[446], 5)

    plot_result(
        restore_patch(X[50].numpy(), (3, 4)),
        restore_patch(Y[50].numpy(), (3, 4)),
        restore_patch(y1, (3, 4))
    )
    
    plot_result(
        restore_patch(X[365].numpy(), (3, 4)),
        restore_patch(Y[365].numpy(), (3, 4)),
        restore_patch(y2, (3, 4))
    )
    
    plot_result(
        restore_patch(X[410].numpy(), (3, 4)),
        restore_patch(Y[410].numpy(), (3, 4)),
        restore_patch(y3, (3, 4))
    )
    
    plot_result(
        restore_patch(X[446].numpy(), (3, 4)),
        restore_patch(Y[446].numpy(), (3, 4)),
        restore_patch(y4, (3, 4))
    )
