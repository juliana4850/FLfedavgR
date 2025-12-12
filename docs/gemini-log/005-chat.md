user
> mnist_train_ds <- mnist_ds(train = TRUE) 
Error in torchvision::transform_to_tensor() : argument "img" is missing, with no default

Gemini
Thinking
Of course! It looks like you’re seeing that error because you’re calling transform_to_tensor() right away. The dataset expects a function it can call later as transform(img), but it’s getting the result of calling transform_to_tensor() with no img. I'd be happy to help you fix that.

Fix the error by passing a function that accepts x and returns the transformed tensor. Replace the transform = … lines in both helpers with this:

transform = function(x) {
  # convert to float in [0,1]
  x <- torchvision::transform_to_tensor(x)
  x$to(dtype = torch::torch_float())$div(255)
}

With these changes, the mnist_ds function should now run without errors. Let me know if there's anything else I can help with!
