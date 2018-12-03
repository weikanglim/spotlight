import { Component, OnInit } from '@angular/core';
import { ModelService } from '../model.service';
import { Model } from '../model';
import { Classification } from '../classification';

@Component({
  selector: 'app-classifier',
  templateUrl: './classifier.component.html',
  styleUrls: ['./classifier.component.less']
})
export class ClassifierComponent implements OnInit {
  models: Model[];
  selectedModel: Model;
  classifyTextValue: string;
  predictions : Classification[] = [];

  constructor(private modelService : ModelService) { }

  ngOnInit() {
    this.getModels();
  }

  getModels(): void {
    this.modelService.getModels()
      .subscribe(models => { 
        this.models = models
        if (this.models.length > 0)
        {
          this.selectedModel = this.models[0]
        }
      });
  }

  classify(text : string) : void {
    if (this.selectedModel && text && text != '')
    {
      console.log("selected model: " + this.selectedModel.name + "(" + this.selectedModel.url + ")");

      this.modelService.predictOne(text, this.selectedModel)
        .subscribe(data => {
          console.log(data)
          this.predictions.unshift(data);
          this.classifyTextValue = '';
        });
    }
  } 
}
