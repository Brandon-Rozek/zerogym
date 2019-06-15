# ZeroMQ interface for OpenAI Gym

This library adds a ZeroMQ interface for the [OpenAI Gym Project](https://github.com/openai/gym). This implementation runs faster than my previous [HTTP interface](https://github.com/Brandon-Rozek/GymHTTP)!

Why would I want to do this? If you want to decouple the processing of 
the environment from the training of your models this might be 
beneficial.

To start the webserver

```bash
python gymserver.py port_num
```

Then in your main application

```python
from gymclient import Environment
env = Environment("127.0.0.1", 5000)
```