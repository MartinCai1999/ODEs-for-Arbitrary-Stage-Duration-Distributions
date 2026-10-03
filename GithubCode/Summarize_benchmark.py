"""Collect every tested setting and select the first passing setting by accuracy."""
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'/'benchmark'
NAMES = ('lognormal','weibull','normal_mixture','normal_lognormal')
METHODS = ('ODE','Trapezoid','Galerkin','MOL')


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    all_rows=[];selected=[]
    for name in NAMES:
        for method in METHODS:
            rows=json.loads((OUT/f'{name}_{method}_raw.json').read_text())
            all_rows.extend(rows)
            for tolerance in (1e-3,1e-5):
                passing=[r for r in rows if r['relative_max_error']<=tolerance]
                # Resolutions were ordered in advance, from coarse to fine.
                # Retain the first setting that reaches the requested error target.
                row=(passing[0] if passing else rows[-1]).copy()
                row.update(tolerance=tolerance,met_tolerance=bool(passing))
                selected.append(row)
    write_csv(OUT/'all_settings.csv',all_rows)
    write_csv(OUT/'selected_settings.csv',selected)
    (OUT/'selected_settings.json').write_text(json.dumps(selected,indent=2))
    for tol in (1e-3,1e-5):
        print('\nTarget normalized error:',tol)
        for name in NAMES:
            print(name,[(r['method'],round(r['cpu_seconds'],6),r['met_tolerance'])
                         for r in selected if r['dataset']==name and r['tolerance']==tol])


if __name__=='__main__':
    main()
