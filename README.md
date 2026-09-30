# Hotel Booking Cancellation Analysis

A reproducible, notebook-first machine-learning project that estimates whether a hotel reservation will be canceled. The positive class is `is_canceled = 1`. The analysis covers validation, transparent cleaning, training-only exploratory analysis, ADR-based business-impact proxies, leakage-safe model comparison, original-column permutation importance, and example scoring.

The outputs are risk estimates learned from historical associations. They are not certainties, causal explanations, or proof that a pricing, deposit, or reminder policy will reduce cancellations.

## Business problem

Hotels plan rooms, staffing, and inventory before guests arrive. Cancellations create uncertainty in that planning process. This project addresses three practical questions:

1. Which booking segments have higher observed cancellation rates?
2. How much booked room-night volume and ADR-based booking value are associated with canceled reservations?
3. Can information available when a reservation is made support a useful cancellation-risk estimate?

## Dataset

- **Source:** [Hotel booking demand on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **File:** `data/hotel_bookings.csv`
- **Observed shape:** 119,390 bookings and 32 source columns
- **Observed arrival period:** July 1, 2015 through August 31, 2017
- **Hotels:** one city hotel and one resort hotel
- **Target:** `is_canceled`, where `1` means canceled and `0` means not canceled
- **Observed target balance:** 44,224 canceled bookings (37.0%) and 75,166 not-canceled bookings (63.0%)
- **File SHA-256:** `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`

## Analysis and modeling approach

The executed notebook:

- verifies the observed file hash, shape, columns, data types, descriptive statistics, missing-value counts and percentages, unique-value counts, categorical cardinality, date ranges, duplicate-looking rows, and target balance;
- keeps the raw CSV unchanged and explains every cleaning decision;
- treats missing `agent` and `company` IDs as structural “no agent” and “no company” states, then replaces the high-cardinality IDs with generalizable binary indicators;
- labels missing `country` values as `Unknown` and leaves missing `children` values for fold-local median imputation;
- flags 180 zero-guest records, one negative ADR value, 715 zero-night records, and 31,994 exact duplicate-looking rows without silently removing records;
- reserves a stratified 80/20 test set with `random_state=42` before exploratory analysis or model selection;
- performs segment analysis and all exploratory plots on the training partition only;
- calculates booked room nights and ADR × booked nights as transparent exposure proxies;
- builds leakage-safe `ColumnTransformer` pipelines with fold-local imputation, one-hot encoding, unseen-category handling, and numeric scaling for Logistic Regression;
- compares a `DummyClassifier`, Logistic Regression, Random Forest, and CatBoost with shuffled five-fold `StratifiedKFold` cross-validation;
- selects by positive-class mean F1, using ROC-AUC only when mean F1 scores are within 0.005;
- evaluates each fitted candidate once on the untouched test set at the default 0.5 threshold;
- interprets the selected pipeline with original-column permutation importance and training-only directional profiles; and
- demonstrates named-column scoring for one example booking.

## Results from the executed notebook

These are descriptive training-partition associations, not causal effects:

- City Hotel bookings canceled at 41.6% across 63,585 training bookings, compared with 28.0% for 31,927 Resort Hotel bookings.
- The 366+ day lead-time band canceled at 67.9% across 2,487 bookings, compared with 9.7% for the 0–7 day band across 15,834 bookings.
- Non Refund deposits had a 99.4% observed cancellation rate across 11,717 bookings. No Deposit bookings canceled at 28.3% across 83,662 bookings. This striking association may reflect selection and policy effects and should not be read causally.
- The Groups market segment canceled at 61.1% across 15,930 bookings.
- Transient bookings canceled at 40.8% across 71,610 bookings, while Transient-Party bookings canceled at 25.4% across 20,217 bookings.

## Business recommendations

1. Review deposit and cancellation policies, especially the selection process around non-refundable bookings, and test changes before broad rollout.
2. Use risk scores to prioritize confirmation reminders for long-lead and high-risk market segments while tracking false-positive costs and customer friction.
3. Incorporate predicted canceled room-night volume into inventory, staffing, and overbooking planning rather than relying only on booking counts.
4. Monitor segment mix, score distributions, precision, recall, and calibration over time.
5. Use controlled experiments before changing pricing, deposits, reminders, or channel strategy. The model ranks risk and does not estimate treatment effects.

## Installation

Python 3.12 is recommended.

### Windows PowerShell

```powershell
git clone https://github.com/BusinessCrab/hotel-booking-cancellation-analysis.git
cd hotel-booking-cancellation-analysis
py -3.12 -m venv ../.venv-hotel-booking
& ../.venv-hotel-booking/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
git clone https://github.com/BusinessCrab/hotel-booking-cancellation-analysis.git
cd hotel-booking-cancellation-analysis
python3.12 -m venv ../.venv-hotel-booking
source ../.venv-hotel-booking/bin/activate
python -m pip install -r requirements.txt
```

## Kaggle download

The CSV is included in this repository for reproducibility. To restore it from the original source after configuring Kaggle credentials, run this command from the repository root:

```bash
kaggle datasets download -d jessemostipak/hotel-booking-demand -p data --unzip
```

The extracted file must be named `data/hotel_bookings.csv`.

## Run the notebook

Start Jupyter from the repository root so the relative data path resolves correctly:

```bash
python -m notebook hotel_booking_cancellation.ipynb
```

Use **Restart Kernel and Run All Cells**. A clean run should execute every cell without manual edits.

## Limitations

- The requested random row split does not demonstrate performance on future bookings. A chronological holdout is the next production-oriented validation step.
- The source has no reservation identifier. Duplicate-looking rows may be legitimate separate bookings, but identical profiles can cross random partitions and make validation optimistic.
- Field timestamps are incomplete. The conservative feature exclusions reduce leakage risk but cannot replace a production data-lineage review.
- Dataset representativeness is uncertain, and hotel, country, channel, pricing, and policy mix may drift.
- The default 0.5 threshold is not cost optimized. Threshold selection should use training-only out-of-fold probabilities and explicit operating costs.
- Probability calibration, subgroup review, drift monitoring, data-quality controls, and scheduled performance monitoring are needed before deployment.
- Observed segment rates and permutation importance are associations. They do not establish causal reasons for cancellation or the effect of an intervention.

## License

The project code is released under the [MIT License](LICENSE). The dataset is not relicensed by this repository and remains subject to its original [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) and attribution requirements.
