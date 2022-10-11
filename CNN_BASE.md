# CNN_BASE / DEBUG の使い方

## **1 `CNN_BASE_DEBUG.py`**

## 1.1`Result` / 結果の出力
**`result フォルダ`** の中に __kernel_shape__ がファイル名に入ったtxtファイルが生成されます。<br>
記述されるのはn個目のデータセットの平均の正解数です。<br>(平均の母数はtestdataすべてを指します。)


<details><summary>result</summary>

```python
result(5,5).txt
result(15,15).txt
```
</details>

<br>

## 1.2 `モデルの保存` 

`dataset_count = 1` にして実行すると、空のh5データを生成し2回目以降はそのデータセットを読み込み、上書き保存を繰り返します。


<details><summary>該当のコード箇所</summary>

```python
38 | if dataset_count == 1:
39 |    load_model = F
40 |    os.remove("saved_model/my_model_88in5.h5")
41 |    empty_file = open("saved_model/my_model_88in5.h5", 'w')
42 |    empty_file.close()
43 |    print("delete past h5 and create new h5 file")
44 |    ...
45 | else:
46 |    load_model = T
```

</details>
<br>

## 1.3 `CNN_BASE_DEBUG.py` の実行前に...
```python
34 | dataset_count = 1
35 | kernel_shape = (10,15)
```
初めに `dataset_count = 1` にして `CNN_BASE_DEBUG.py` を実行してください。

<br>

## **2 `CNN_BASE_DEBUG.py`**
## 2.1 パラメータの調整
**`end`** を調整してください。<br>
`end個` の dataset を用いて学習を行います。
<br>

## 2.2 実行
### 1.3　を設定したうえで実行してください。