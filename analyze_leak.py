import pandas as pd
df = pd.read_csv('leak_val.csv')
print('Total val checked:', len(df))
h10 = (df['hamming'] <= 10).sum()
print(f'Distance <= 10: {h10} ({h10/len(df)*100:.1f}%)')
h6 = (df['hamming'] <= 6).sum()
print(f'Distance <= 6: {h6} ({h6/len(df)*100:.1f}%)')
h0 = (df['hamming'] == 0).sum()
print(f'Distance == 0: {h0} ({h0/len(df)*100:.1f}%)')

is_syn = df['val_image'].str.contains('empty_syn')
print('Synthetic empty in val:', is_syn.sum())
print('Synthetic empty <= 10:', (is_syn & (df['hamming'] <= 10)).sum())

real_leak = df[~is_syn & (df['hamming'] <= 10)]
print('Real images <= 10:', len(real_leak))
for _, r in real_leak.head(15).iterrows():
    v = r['val_image'].replace('\\', '/').split('/')[-1]
    t = r['nearest_train_image'].replace('\\', '/').split('/')[-1]
    d = r['hamming']
    print(f'  d={d:2d}  {v}  <->  {t}')

