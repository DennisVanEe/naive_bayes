use anyhow::{Context, Result};
use csv::{Reader, StringRecord};
use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    path::Path,
};

pub struct NaiveBayesClassifier {
    disease_betas: HashMap<String, HashMap<String, f64>>,
    disease_pis: HashMap<String, f64>,
}

impl NaiveBayesClassifier {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut reader = csv::Reader::from_path(path)?;

        // Collect all disease information:

        let mut diseases_map = HashMap::new();
        let mut all_symptoms = HashSet::new();
        let mut num_records = 0;
        for record in reader.records() {
            let record = record?;
            let disease = record.get(0).context("csv record missing disease entry.")?;

            let mut symptoms = HashSet::new();
            for symptom in record.iter().skip(1) {
                if symptom.is_empty() {
                    continue;
                }
                symptoms.insert(symptom.trim().to_string());
                all_symptoms.insert(symptom.trim().to_string());
            }

            match diseases_map.entry(disease.to_string()) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(Vec::new()),
            }
            .push(symptoms);

            num_records += 1;
        }

        // Calculate all of the beta values:

        let total_num_symptoms = all_symptoms.len() as f64; // N
        let mut disease_betas = HashMap::new();
        for (disease, symptoms_instances) in &diseases_map {
            // Get total number of symptoms for this disease:
            let num_symptoms = symptoms_instances
                .iter()
                .fold(0, |acc, symptoms| acc + symptoms.len())
                as f64;

            // Now, for each symptom, we calculate the beta value:
            let mut betas = HashMap::new();
            for symptom in &all_symptoms {
                // Count how often this occurs for this disease:
                let num_symptom = symptoms_instances.iter().fold(0, |acc, symptoms| {
                    if symptoms.contains(symptom) {
                        acc + 1
                    } else {
                        acc
                    }
                }) as f64;

                let beta = (num_symptom + 1.0) / (num_symptoms + total_num_symptoms);
                betas.insert(symptom.clone(), beta);
            }

            disease_betas.insert(disease.clone(), betas);
        }

        // Calculate all of the pi values:
        let mut disease_pis = HashMap::new();
        for (disease, symptoms_instances) in &diseases_map {
            let pi = (symptoms_instances.len() as f64) / (num_records as f64);
            disease_pis.insert(disease.clone(), pi);
        }

        Ok(NaiveBayesClassifier {
            disease_betas,
            disease_pis,
        })
    }

    // Predicts a bunch of values from a test.csv path:
    pub fn predict<P: AsRef<Path>>(&self, inpath: P, outpath: P) -> Result<()> {
        let mut reader = csv::Reader::from_path(inpath)?;

        let mut results = Vec::new();
        for record in reader.records() {
            let record = record?;

            let mut symptoms = HashSet::new();
            for symptom in record.iter().skip(1) {
                if symptom.is_empty() {
                    continue;
                }
                symptoms.insert(symptom.trim().to_string());
            }

            results.push(self.predict_one(&symptoms));
        }

        // Now we can write the result:
        let mut writer = csv::Writer::from_path(outpath)?;

        writer.write_record(&["ID", "Disease"])?;
        for (i, result) in results.iter().enumerate() {
            writer.write_record(&[(i + 1).to_string(), result.to_string()])?;
        }

        Ok(())
    }

    /// Given a record of symptoms, makes a prediction as to which disease it is:
    fn predict_one(&self, psymptoms: &HashSet<String>) -> &str {
        let (best_disease, _) = self.disease_betas.iter().fold(
            ("", -1.0),
            |(best_disease, best_score), (disease, betas)| {
                let product_betas = psymptoms.iter().fold(1.0, |acc, psymptom| {
                    let &beta = betas.get(psymptom).unwrap(); // this should always succeed.
                    acc * beta
                });

                let pi = self.disease_pis.get(disease).unwrap();
                let score = pi * product_betas;

                if score > best_score {
                    (&disease, score)
                } else {
                    (best_disease, best_score)
                }
            },
        );

        best_disease
    }
}
