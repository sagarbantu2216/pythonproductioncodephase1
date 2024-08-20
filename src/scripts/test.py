# Input dictionary
input_data = {
  "Problem List": [
    "Vitamin D deficiency",
    "Recurrent major depression",
    "Generalized anxiety disorder",
    "Binge eating disorder",
    "Attention deficit hyperactivity disorder",
    "Neuropathy",
    "Peripheral motor neuropathy",
    "Nuclear cataract",
    "Essential hypertension",
    "Gastroesophageal reflux disease",
    "Left inguinal hernia",
    "Paraesophageal hernia",
    "Chronic cystitis",
    "Foot callus",
    "Chronic low back pain",
    "History of urinary tract infection",
    "Acquired hammer toe of right foot",
    "Partial amputation of right toe",
    "History of injury of eye region"
  ],
  "metoprolol succinate ER 50 mg tablet,extended release 24 hr": "Take 1 tablet every day by oral route for 28 days.",
  "lisinopril 20 mg tablet": "Take 1 tablet every day by oral route as directed for 28 days.",
  "dextroamphetamine-amphetamine 10 mg tablet": "Take 1 tablet twice a day by oral route as directed for 28 days.",
  "Adderall 10 mg tablet": "Take 1 tablet twice a day by oral route as directed for 28 days.",
  "sulfamethoxazole 800 mg-trimethoprim 160 mg tablet": "TAKE 1 TABLET BY MOUTH TWICE A DAY FOR 14 DAYS",
  "Bactrim DS 800 mg-160 mg tablet": "Take 1 tablet every 12 hours by oral route for 7 days.",
  "hydrocodone 10 mg-acetaminophen 325 mg tablet": "Take 1 tablet 4 times a day by oral route as needed for 14 days.",
  "acetaminophen ER 650 mg tablet,extended release": "TAKE 1 TABLET BY MOUTH EVERY 8 HOURS AS NEEDED FOR 28 DAYS",
  "Tylenol Arthritis Pain 650 mg tablet,extended release": "Take 1 tablet every 8 hours by oral route as needed for 28 days.",
  "ciclopirox 8 % topical solution": "APPLY TO THE AFFECTED AREA(S) BY TOPICAL ROUTE ONCE DAILY PREFERABLY AT BEDTIME OR 8 HOURS BEFORE WASHING",
  "lorazepam 0.5 mg tablet": "TAKE 1 TABLET EVERY DAY BY ORAL ROUTE AS NEEDED.",
  "cephalexin 500 mg capsule": "TAKE 1 CAPSULE BY MOUTH FOUR TIMES A DAY FOR 14 DAYS",
  "pantoprazole 40 mg tablet,delayed release": "Take 1 tablet every day by oral route in the morning for 28 days.",
  "furosemide 20 mg tablet": "Take 1 tablet every day by oral route in the morning for 28 days.",
  "chlorhexidine gluconate 0.12 % mouthwash": "RINSE AND SPIT- TWICE A DAY W/10 ML FOR PLAQUE CONTROL OR SWAB GUMLINES NIGHTLY AND DONT WASH AWAY RESIDUE",
  "Vitamin D3 50 mcg (2,000 unit) capsule": "Take 1 capsule every day by oral route as directed for 28 days.",
  "Salonpas 3.1 %-10 %-6 % topical patch": "Apply 1 patch 4 times a day by topical route as needed."
}

# Convert the input dictionary to the desired format
output_data = {
    "problems": input_data["Problem List"]
}

for medication, instruction in input_data.items():
    if medication != "Problem List":
        output_data["problems"].append(f"{medication} {instruction}")

# Print the converted output
import json
print(json.dumps(output_data, indent=2))
