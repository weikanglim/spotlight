import { Component, OnInit } from '@angular/core';
import { UploadEvent, UploadFile, FileSystemFileEntry, FileSystemDirectoryEntry } from 'ngx-file-drop';
import { ModelFile } from '../modelfile';
import { ModelService } from '../model.service';
import { Classifier } from '../classifier';
import { Router } from '@angular/router';
import { ClassifierService } from '../classifier.service';
import { Model } from '../model';

enum InputType {
    Classifier,
    Model,
    ModelFile,
    None
}

@Component({
  selector: 'app-train',
  templateUrl: './train.component.html',
  styleUrls: ['./train.component.less']
})
export class TrainComponent implements OnInit {
  classifiers : Classifier[];
  selectedClassifier: Classifier;

  models: Model[];
  selectedModel: Model;

  modelFile: File;
  dataFile : File;
  name: string = "";
  error: string;
  info: string;

  constructor(public modelService: ModelService, public classifierService: ClassifierService, public router : Router) {}

  ngOnInit() {
    this.getClassifiers();
    this.getModels();
  }

  private getClassifiers(): void {
    this.classifierService.getClassifiers()
      .subscribe(results => { 
        this.classifiers = results
      });
  }

  private getModels(): void {
    this.modelService.getModels()
      .subscribe(models => { 
        this.models = models
      });
  }

  private getInputType(): InputType {
    console.log(this.selectedModel);
    console.log(!this.selectedClassifier);
    if (this.selectedClassifier) return InputType.Classifier;
    if (this.selectedModel) return InputType.Model;
    if (this.modelFile) return InputType.ModelFile;

    return InputType.None;
  }

  private getInputName() : string
  {
    switch (this.getInputType())
    {
      case InputType.Classifier:
        return this.selectedClassifier.name;
      
      case InputType.Model:
        return this.selectedModel.name;

      case InputType.ModelFile:
        return this.getFileName(this.modelFile);

      default:
        return "";
    }
  }

  private getFileName(file: File) : string
  {
    var fileName = file.name;
    var extensionIndex = fileName.lastIndexOf(".");

    if (extensionIndex > 0)
    {
      fileName = fileName.substring(0, extensionIndex)
    }

    return fileName;
  }

  private updateName() : void {
    if (this.dataFile && this.name == "" && this.getInputName() != "")
    {
      this.name = this.getInputName() + "_" + this.getFileName(this.dataFile)
    }
  }

  public modelFileDropped(event: UploadEvent) {
    for (const droppedFile of event.files) {
      if (droppedFile.fileEntry.isFile) {
        const fileEntry = droppedFile.fileEntry as FileSystemFileEntry;
        fileEntry.file((file: File) => {
          console.log(droppedFile.relativePath, file);

          this.modelFile = file;
          this.updateName();
        });
      }
    }
  }

  public dataFileDropped(event: UploadEvent) {
    for (const droppedFile of event.files) {
      if (droppedFile.fileEntry.isFile) {
        const fileEntry = droppedFile.fileEntry as FileSystemFileEntry;
        fileEntry.file((file: File) => {
          console.log(droppedFile.relativePath, file);

          this.dataFile = file;
          this.updateName();
        });
      }
    }
  }

  private validate() : boolean {
    if (this.getInputType() == InputType.None)
    {
      this.error = "Please select a classifier, an existing trained model, or upload a trained model."
      return false;
    }

    if (!this.dataFile)
    {
      this.error = "Please upload training data."
      return false;
    }

    if (!this.name)
    {
      this.error = "Please provide a name for the model."
      return false;
    }

    return true;
  }


  private handleError() : void {
    console.log("Error handled")
    this.error = "Unable to complete your request. Please try again.";
  }

  public onSubmit() : void {
    this.error = null;

    if (!this.validate())
    {
      return;
    }
    
    this.info = "Training classifier...";

    if (this.getInputType() == InputType.Classifier)
    {
      this.classifierService.train(this.selectedClassifier, this.dataFile, this.name).subscribe(data => {
        this.router.navigateByUrl(this.router.createUrlTree(
          ['/'], {queryParams: {"savedModel":  this.name }}
        ));
      },
      err => this.handleError);
    }
    else if (this.getInputType() == InputType.Model)
    {
      this.modelService.trainExisting(this.selectedModel, this.dataFile, this.name).subscribe(data => {
        this.router.navigateByUrl(this.router.createUrlTree(
          ['/'], {queryParams: {"savedModel":  this.name }}
        ));
      },
      err => this.handleError);
    }
    else
    {
      this.modelService.train(this.modelFile, this.dataFile, this.name).subscribe(data => {
        this.router.navigateByUrl(this.router.createUrlTree(
          ['/'], {queryParams: {"savedModel":  this.name }}
        ));
      },
      err => this.handleError);
    }
    
  }
}
