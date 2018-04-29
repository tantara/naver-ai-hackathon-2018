# Naver AI Hackathon 2018 - Movie Prediction

I participated in [Naver AI Hackathon 2018](https://github.com/naver/ai-hackathon-2018) and **ranked 5th(over 200 teams, MSE: 2.76661)** as an inidivisual participant(Team: tantara).

#### Final Leaderboard

![learderboard](docs/learderboard.png)



## Features

- Fully dockerized environments
- Tokenization with `Twitter` POS Tagger
- Stacked LSTM
- Bidirectional LSTM
- CNN + LSTM(but not used)





## How to Use

#### on Docker

Install `docker` first, and run commands as follows:

```bash
$ sh build.sh && sh run.sh
$ cd movie-predication
$ python lstm.py --epochs 30 --lr 0.001 --num_layers 4 --embedding 300 --hidden_dim 500 --batch 4000 --dr_embed 0.5
```



#### on NSML

Login with `nsml` first, and run commands as follows:

````bash
$ cd movie-prediction
$ nsml run -d movie_final -e lstm.py -a "--epochs 30 --lr 0.001 --num_layers 4 --embedding 300 --hidden_dim 500 --batch 4000 --dr_embed 0.5"
````



## Hyperparameters

|           Name            |   Value    |                            |
| :-----------------------: | :--------: | -------------------------- |
|      Training Epoch       |     30     | --epochs                   |
|       Learning Rate       |   0.001    | --lr                       |
|        Batch Size         |    4000    | --batch, *It depends on GPU* |
|       Drouout Rate        |    0.5     | --dropout_rate, --dr_embed   |
|     the size of vocab     |  110,000   | *Pretrained*               |
|   the number of tokens    |  Up to 20  | --strmaxlen, *Zero padding* |
|      Word Embedding       |    300     | --embedding                 |
| the number of LSTM layers |     4      | --num_layers                |
|    LSTM's hidden size     |    300     | --hidden                    |
|       Bidirectional       | True/False | --bi, *Optional*            |
|            CNN            | True/False | --cnn, *Not Used*           |

## Results

* tantara/movie_final/24/xx: Stacked LSTM
  * Top Score: 2.77174
* tantara/movie_final/32/xx: Stacked LSTM + Bidirectional
  * **Top Score: 2.76661**

![results](docs/results.png)

## Reference

1. [Naver AI's Official Baseline](https://github.com/naver/ai-hackathon-2018/tree/master/missions/examples/movie-review)
2. [jiangqy/LSTM-Classification-Pytorch](https://github.com/jiangqy/LSTM-Classification-Pytorch)

## License

```
Copyright 2018 tantara.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
