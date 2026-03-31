import pandas as pd
import numpy as np
import os

INPUT_FILES = {
    "ARP_Spoofing":      "ARP_Spoofing_train_pcap.csv",
    "Recon-Port_Scan":   "Recon-Port_Scan_train_pcap.csv",
    "TCP_IP-DoS-ICMP1":  "TCP_IP-DoS-ICMP1_train_pcap.csv",
}

SELECTED_FEATURES = [
    "Protocol_Type",
    "TCP", "UDP", "ICMP", "ARP",
    "syn_flag_number", "rst_flag_number", "ack_flag_number",
    "psh_flag_number", "syn_count", "fin_count",
    "Rate", "Duration", "Header_Length",
    "Max", "Variance", "Magnitude",
    "HTTPS",
    "attack_type",
]


def clean_file(label, path):
    df = pd.read_csv(path)

    df.drop_duplicates(inplace=True)

    zero_var_cols = [c for c in df.columns if df[c].std() == 0]
    df.drop(columns=zero_var_cols, inplace=True)

    for col in ["Rate", "Srate"]:
        if col in df.columns:
            df[col] = df[col].clip(upper=df[col].quantile(0.999))

    df = df[df["Rate"] > 0].reset_index(drop=True)

    if "Header_Length" in df.columns:
        df["Header_Length"] = df["Header_Length"].clip(upper=df["Header_Length"].quantile(0.99))

    if "Duration" in df.columns:
        df["Duration"] = df["Duration"].clip(upper=df["Duration"].quantile(0.999))

    df["attack_type"] = label

    return df


def merge_and_select(cleaned_dfs):
    merged = pd.concat(cleaned_dfs.values(), ignore_index=True)

    merged.rename(columns={
        "Protocol Type": "Protocol_Type",
        "Magnitue":      "Magnitude",
    }, inplace=True)

    return merged[SELECTED_FEATURES].copy()


def main():
    cleaned = {}

    for label, filename in INPUT_FILES.items():
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Input file not found: {filename}")
        df_clean = clean_file(label, filename)
        out_path = filename.replace("_train_pcap.csv", "_cleaned.csv")
        df_clean.to_csv(out_path, index=False)
        cleaned[label] = df_clean

    df_final = merge_and_select(cleaned)
    df_final.to_csv("iomt_merged_clean.csv", index=False)

    print(f"Done. Shape: {df_final.shape[0]:,} rows x {df_final.shape[1]} cols")


if __name__ == "__main__":
    main()
