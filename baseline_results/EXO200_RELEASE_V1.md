# EXO-200 baseline results are from open-data release v1

`classification/exo200/` and `regression/exo200/` were produced on release v1 (300-sample waveforms, no event blacklist). The `exo200` reader now implements release v3
(`validation_skill.md` card 6.2: 600-sample waveforms, blacklisted events removed, different split sizes and class balance). These files are kept for the record; they are not
comparable with runs on release v3 and have to be regenerated. The other datasets are unaffected.
