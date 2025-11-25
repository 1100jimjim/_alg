```text
PS C:\ccc\py2cs> & C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe c:/ccc/py2cs/HW/HW8.py
請輸入字串1：ATGCAATCCC
請輸入字串2：ATG A TCCG

「ATGCAATCCC」 變成 「ATG A TCCG」 的最小編輯距離為：3

DP 表格如下（可用來觀察動態規劃過程）：
       A  T  G     A     T  C  C  G
    0  1  2  3  4  5  6  7  8  9 10
 A  1  0  1  2  3  4  5  6  7  8  9
 T  2  1  0  1  2  3  4  5  6  7  8
 G  3  2  1  0  1  2  3  4  5  6  7
 C  4  3  2  1  1  2  3  4  4  5  6
 A  5  4  3  2  2  1  2  3  4  5  6
 A  6  5  4  3  3  2  2  3  4  5  6
 T  7  6  5  4  4  3  3  2  3  4  5
 C  8  7  6  5  5  4  4  3  2  3  4
 C  9  8  7  6  6  5  5  4  3  2  3
 C 10  9  8  7  7  6  6  5  4  3  3
```
