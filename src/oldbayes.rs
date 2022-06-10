use anyhow::{Context, Result};
use csv::{Reader, StringRecord};
use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    path::Path,
};

pub struct NaiveBayesClassifier {
    diseases: Vec<(String, Vec<Vec<bool>>)>,
    disease_probs: Vec<f64>,
    symptom_order: Vec<String>,
}

impl NaiveBayesClassifier {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut reader = csv::Reader::from_path(path)?;

        // Sort into diseases:
        let mut diseases_map = HashMap::new();
        let mut all_symptoms = HashSet::new();
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
        }

        // Get an ordering for all symptoms that we will use:
        let mut symptom_order = Vec::with_capacity(all_symptoms.len());
        for symptom in &all_symptoms {
            symptom_order.push(symptom.clone());
        }
        symptom_order.sort();

        let diseases = diseases_map
            .iter()
            .map(|(disease, symptom_instances)| {
                let symptoms = symptom_instances
                    .iter()
                    .map(|symptoms| Self::symptom_vector(&symptom_order, symptoms))
                    .collect();
                (disease.clone(), symptoms)
            })
            .collect();

        // Collect the probability that it is one of the given diseases:
        let mut disease_probs: Vec<f64> = diseases_map
            .iter()
            .map(|(_, symptom_instances)| symptom_instances.len() as f64)
            .collect();
        let total_entries: f64 = disease_probs.iter().sum();
        disease_probs.iter_mut().for_each(|p| *p /= total_entries);

        Ok(NaiveBayesClassifier {
            diseases,
            disease_probs,
            symptom_order,
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

    fn symptom_vector(symptom_order: &[String], symptoms: &HashSet<String>) -> Vec<bool> {
        symptom_order
            .iter()
            .map(|symptom| symptoms.contains(symptom))
            .collect()
    }

    /// Given a record of symptoms, makes a prediction as to which disease it is:
    fn predict_one(&self, psymptoms: &HashSet<String>) -> &str {
        let psymptoms = Self::symptom_vector(&self.symptom_order, psymptoms);

        // calculate probability for each individual symptom state:
        let mut symptom_counts = vec![0; psymptoms.len()];

        let mut highest_prob = -1.0; //0.0;
        let mut best_disease = "";
        for ((disease, symptom_instances), &disease_prob) in
            self.diseases.iter().zip(self.disease_probs.iter())
        {
            // now we perform P(s|d):
            for symptoms in symptom_instances {
                for (count, (&psymptom, &symptom)) in symptom_counts
                    .iter_mut()
                    .zip(psymptoms.iter().zip(symptoms.iter()))
                {
                    if psymptom == symptom {
                        *count += 1;
                    }
                }
            }

            // Now we can calculate some probabilities:
            let prob = psymptoms.iter().zip(symptom_counts.iter()).fold(
                1.0,
                |acc, (has_symptom, count)| {
                    if *has_symptom {
                        acc * ((*count as f64) / (symptom_instances.len() as f64))
                    } else {
                        acc
                    }
                },
            ) * disease_prob;

            if prob > highest_prob {
                highest_prob = prob;
                best_disease = &disease;
            }

            symptom_counts.fill(0);
        }

        best_disease
    }
}
