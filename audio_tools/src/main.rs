use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::{Arc, Mutex};
use eframe::{egui, App, Frame};
use egui::{Color32, Context, Pos2, Rangef, Rect, Shape, Slider};
use egui_plot::{Line, Plot};
use pitch_detection::detector::mcleod::McLeodDetector;
use pitch_detection::detector::PitchDetector;

const FFT_SIZE: usize = 2048;

struct SharedAudioBuffer {
    samples: Vec<f32>,
}

impl SharedAudioBuffer {
    fn new() -> Self {
        Self {
            samples: vec![0.0; FFT_SIZE],
        }
    }

    fn update(&mut self, new_samples: &[f32]) {
        let len = self.samples.len();
        let ns_len = new_samples.len();

        if ns_len >= len {
            self.samples.copy_from_slice(&new_samples[ns_len - len..]);
        } else {
            self.samples.rotate_left(ns_len);
            self.samples[len - ns_len..].copy_from_slice(new_samples);
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum FreqRangePreset {
    All,
    Low,
    Mid,
    High,
}

impl FreqRangePreset {
    fn label(&self) -> &'static str {
        match self {
            FreqRangePreset::All => "Tout (0–Nyquist)",
            FreqRangePreset::Low => "Graves (0–500 Hz)",
            FreqRangePreset::Mid => "Médiums (500–2000 Hz)",
            FreqRangePreset::High => "Aigus (2000+ Hz)",
        }
    }
}

struct AudioApp {
    audio_buffer: Arc<Mutex<SharedAudioBuffer>>,
    fft_output: Vec<f32>,
    planner: FftPlanner<f32>,
    sample_rate: f32,

    waveform_zoom: f32,
    fft_zoom: f32,
    speed_factor: f32,
    detector: McLeodDetector<f32>,

    custom_range_active: bool,
    custom_range: Rangef,
    freq_preset: FreqRangePreset
}

impl AudioApp {
    fn new(audio_buffer: Arc<Mutex<SharedAudioBuffer>>, sample_rate: f32) -> Self {
        Self {
            audio_buffer,
            fft_output: vec![0.0; FFT_SIZE / 2],
            planner: FftPlanner::new(),
            sample_rate,
            waveform_zoom: 1.0,
            fft_zoom: 1.0,
            speed_factor: 1.0,
            detector: McLeodDetector::new(FFT_SIZE, FFT_SIZE / 2),
            custom_range_active: false,
            custom_range: Rangef::new(0.0, sample_rate / 2.0),
            freq_preset: FreqRangePreset::All,
        }
    }

    fn process_fft(&mut self) {
        let audio_buf = self.audio_buffer.lock().unwrap();
        let mut buffer: Vec<Complex<f32>> = audio_buf.samples.iter()
            .map(|&f| Complex { re: f, im: 0.0 })
            .collect();

        let fft = self.planner.plan_fft_forward(FFT_SIZE);
        fft.process(&mut buffer);

        for i in 0..FFT_SIZE / 2 {
            self.fft_output[i] = buffer[i].norm();
        }
    }

    fn detect_pitch(&mut self) -> Option<(f32, String)> {
        let audio_buf = self.audio_buffer.lock().unwrap();
        let samples = &audio_buf.samples;

        if let Some(pitch) = self.detector.get_pitch(samples, self.sample_rate as usize, 0.93, 0.1) {
            let note = freq_to_note_name(pitch.frequency);
            Some((pitch.frequency, note))
        } else {
            None
        }
    }
}

impl App for AudioApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        self.process_fft();

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Zoom Onde :");
                ui.add(Slider::new(&mut self.waveform_zoom, 0.1..=5.0).logarithmic(true));
                ui.label("Zoom FFT :");
                ui.add(Slider::new(&mut self.fft_zoom, 0.1..=5.0).logarithmic(true));
                ui.label("Vitesse :");
                ui.add(Slider::new(&mut self.speed_factor, 0.1..=3.0));
                ui.checkbox(&mut self.custom_range_active, "Activer plage personnalisée");
                if self.custom_range_active {
                    let nyquist = self.sample_rate / 2.0;
                    ui.add(Slider::new(&mut self.custom_range.min, 0.0..=self.custom_range.max).text("Min Hz"));
                    ui.add(Slider::new(&mut self.custom_range.max, self.custom_range.min..=nyquist).text("Max Hz"));
                }

                ui.label("Plage de fréquences :");
                egui::ComboBox::from_id_source("freq_preset")
                    .selected_text(self.freq_preset.label())
                    .show_ui(ui, |ui| {
                        for preset in [
                            FreqRangePreset::All,
                            FreqRangePreset::Low,
                            FreqRangePreset::Mid,
                            FreqRangePreset::High,
                        ] {
                            ui.selectable_value(&mut self.freq_preset, preset, preset.label());
                        }
                    });
            });

            ui.separator();

            egui::CollapsingHeader::new("Forme d'onde").default_open(true).show(ui, |ui| {
                let audio_buf = self.audio_buffer.lock().unwrap();
                let samples = &audio_buf.samples;

                let (min, max) = samples.iter().fold(
                    (f32::INFINITY, f32::NEG_INFINITY),
                    |(min, max), &v| (min.min(v), max.max(v)),
                );
                let avg_amplitude = samples.iter().map(|v| v.abs()).sum::<f32>() / samples.len() as f32;
                let rms = (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32).sqrt();
                let db = 20.0 * rms.max(1e-6).log10();

                ui.label(format!("Max: {:.2}, Min: {:.2}, Moy abs: {:.2}, Volume: {:.1} dB", max, min, avg_amplitude, db));

                let waveform_points: Vec<[f64; 2]> = samples.iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64, (v * self.waveform_zoom) as f64])
                    .collect();

                Plot::new("waveform")
                    .height(300.0)
                    .allow_zoom(true)
                    .allow_scroll(true)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("waveform", waveform_points));
                    });
            });

            ui.separator();

            egui::CollapsingHeader::new("Spectre FFT").default_open(true).show(ui, |ui| {
                let (max_idx, max_val) = self.fft_output
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap_or((0, &0.0));

                let freq_resolution = self.sample_rate / FFT_SIZE as f32;
                let peak_freq = max_idx as f32 * freq_resolution;
                let avg_energy = self.fft_output.iter().copied().sum::<f32>() / self.fft_output.len() as f32;

                ui.label(format!("Pic: {:.1} Hz, Intensité: {:.2}, Energie Moy: {:.2}",
                                 peak_freq, max_val, avg_energy));
                let (start_idx, end_idx) = if self.custom_range_active {
                    let nyquist = self.sample_rate / 2.0;
                    let start = (self.custom_range.min / freq_resolution).clamp(0.0, nyquist) as usize;
                    let end = (self.custom_range.max / freq_resolution).clamp(0.0, nyquist) as usize;
                    (start.min(end), end.max(start).min(self.fft_output.len()))
                } else {
                    match self.freq_preset {
                        FreqRangePreset::All => (0, self.fft_output.len()),
                        FreqRangePreset::Low => {
                            let end = (500.0 / freq_resolution).min(self.fft_output.len() as f32) as usize;
                            (0, end)
                        }
                        FreqRangePreset::Mid => {
                            let start = (500.0 / freq_resolution) as usize;
                            let end = (2000.0 / freq_resolution).min(self.fft_output.len() as f32) as usize;
                            (start, end)
                        }
                        FreqRangePreset::High => {
                            let start = (2000.0 / freq_resolution).min(self.fft_output.len() as f32) as usize;
                            (start, self.fft_output.len())
                        }
                    }
                };

                ui.label(format!("Plage affichée : {:.0} Hz → {:.0} Hz",
                                 start_idx as f32 * freq_resolution,
                                 end_idx as f32 * freq_resolution));

                let spectrum_points: Vec<[f64; 2]> = self.fft_output[start_idx..end_idx]
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let freq = (start_idx + i) as f32 * freq_resolution;
                        [freq as f64, (v.log10().max(-3.0) + 3.0) as f64]
                    })
                    .collect();

                Plot::new("spectrum")
                    .height(200.0)
                    .allow_scroll(true)
                    .allow_zoom(true)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("spectrum", spectrum_points));
                    });

                if let Some((freq, note)) = self.detect_pitch() {
                    ui.label(format!("Note dominante: {} ({:.1} Hz)", note, freq));
                }
            });
        });

        ctx.request_repaint_after(std::time::Duration::from_millis((16.0 / self.speed_factor) as u64));
    }
}

fn freq_to_note_name(freq: f32) -> String {
    let a4 = 440.0;
    let semitones_from_a4 = (12.0 * (freq / a4).log2()).round();
    let midi_note = 69.0 + semitones_from_a4;
    let note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    let name = note_names[midi_note as usize % 12];
    let octave = (midi_note as i32 / 12) - 1;
    format!("{}{}", name, octave)
}

fn main() -> Result<(), anyhow::Error> {
    let audio_buffer = Arc::new(Mutex::new(SharedAudioBuffer::new()));
    let audio_buffer_clone = audio_buffer.clone();

    let host = cpal::default_host();
    let device = host.default_input_device().expect("Pas de micro détecté");
    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0 as f32;

    let err_fn = |err| eprintln!("Erreur audio: {}", err);

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buf = audio_buffer_clone.lock().unwrap();
            buf.update(data);
        },
        err_fn,
        None,
    )?;

    stream.play()?;

    let app = AudioApp::new(audio_buffer, sample_rate);
    let native_options = eframe::NativeOptions::default();
    let _ = eframe::run_native("MEOW ! ᓚᘏᗢ", native_options, Box::new(|_cc| Ok(Box::new(app))));

    Ok(())
}