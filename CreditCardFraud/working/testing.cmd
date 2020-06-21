pushd ..\input\creditcardzip\
cp results_header.csv results.csv
popd

for /l %%x in (1, 1, 100) do (
   echo %%x
   python CreditCardFraud.py
)
